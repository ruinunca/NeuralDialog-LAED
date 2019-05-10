# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
import torch
import torch.nn as nn
import torch.nn.functional as F
from laed.dataset.corpora import PAD, BOS, EOS
from torch.autograd import Variable
from laed import criterions
from laed.enc2dec.decoders import DecoderRNN
from laed.enc2dec.encoders import EncoderRNN, RnnUttEncoder
from laed.utils import INT, FLOAT, LONG, cast_type
from laed import nn_lib
from laed.models.model_bases import BaseModel
from laed.enc2dec.decoders import GEN, TEACH_FORCE
from laed.utils import Pack
import itertools
import numpy as np


class LAED(BaseModel):
    def qzx_forward(self, out_utts):
        # output encoder
        output_embedding = self.x_embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        x_last = x_last.transpose(0, 1).contiguous().view(-1, self.config.dec_cell_size)
        qy_logits = self.q_y(x_last).view(-1, self.config.k)

        # switch that controls the sampling
        if self.kl_w == 1.0 and self.config.greedy_q:
            sample_y, y_ids = self.greedy_cat_connector(qy_logits, self.use_gpu,
                                                        return_max_id=True)
        else:
            sample_y, y_ids = self.cat_connector(qy_logits, 1.0, self.use_gpu,
                                                 hard=not self.training, return_max_id=True)

        sample_y = sample_y.view(-1, self.config.k * self.config.y_size)
        y_ids = y_ids.view(-1, self.config.y_size)

        return Pack(qy_logits=qy_logits, sample_y=sample_y, y_ids=y_ids)

    def exp_forward(self, data_feed):
        ctx_lens = data_feed['context_lens']
        batch_size = len(ctx_lens)

        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        output_lens = self.np2var(data_feed['output_lens'], FLOAT)

        # context encoder
        c_inputs = self.utt_encoder(ctx_utts)
        c_outs, c_last = self.ctx_encoder(c_inputs, ctx_lens)
        c_last = c_last.squeeze(0)

        # prior network
        py_logits = self.p_y(c_last).view(-1, self.config.k)
        log_py = F.log_softmax(py_logits, dim=py_logits.dim()-1)

        exp_size = np.power(self.config.k, self.config.y_size)
        sample_y = cast_type(
            Variable(torch.zeros((exp_size * self.config.y_size, self.config.k))), FLOAT, self.use_gpu)
        d = dict((str(i), range(self.config.k)) for i in range(self.config.y_size))
        all_y_ids = []
        for combo in itertools.product(*[d[k] for k in sorted(d.keys())]):
            all_y_ids.append(list(combo))
        np_y_ids = np.array(all_y_ids)
        np_y_ids = self.np2var(np_y_ids, LONG)
        # map sample to initial state of decoder
        sample_y.scatter_(1, np_y_ids.view(-1, 1), 1.0)
        sample_y = sample_y.view(-1, self.config.k * self.config.y_size)

        # pack attention context
        attn_inputs = None
        labels = out_utts[:, 1:].contiguous()
        c_last = c_last.unsqueeze(0)

        nll_xcz = 0.0
        cum_pcs = 0.0
        all_words = torch.sum(output_lens-1)
        for exp_id in range(exp_size):
            cur_sample_y = sample_y[exp_id:exp_id+1]
            cur_sample_y = cur_sample_y.expand(batch_size, self.config.k*self.config.y_size)

            # find out logp(z|c)
            log_pyc = torch.sum(log_py.view(-1, self.config.k*self.config.y_size) * cur_sample_y, dim=1)
            # map sample to initial state of decoder
            dec_init_state = self.c_init_connector(cur_sample_y) + c_last

            # decode
            dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                       out_utts[:, 0:-1],
                                                       dec_init_state,
                                                       attn_context=attn_inputs,
                                                       mode=TEACH_FORCE, gen_type="greedy",
                                                       beam_size=self.config.beam_size)

            output = dec_outs.view(-1, dec_outs.size(-1))
            target = labels.view(-1)
            enc_dec_nll = F.nll_loss(output, target, size_average=False,
                                     ignore_index=self.nll_loss.padding_idx,
                                     weight=self.nll_loss.weight, reduce=False)

            enc_dec_nll = enc_dec_nll.view(-1, dec_outs.size(1))
            enc_dec_nll = torch.sum(enc_dec_nll, dim=1)
            py_c = torch.exp(log_pyc)
            cum_pcs += py_c
            nll_xcz += py_c * enc_dec_nll

        nll_xcz = torch.sum(nll_xcz) / all_words
        return Pack(nll=nll_xcz)

    def greedy_forward(self, data_feed):
        ctx_lens = data_feed['context_lens']
        batch_size = len(ctx_lens)
        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # context encoder
        c_inputs = self.utt_encoder(ctx_utts)
        c_outs, c_last = self.ctx_encoder(c_inputs, ctx_lens)
        c_last = c_last.squeeze(0)

        # prior network
        py_logits = self.p_y(c_last).view(-1, self.config.k)

        # map sample to initial state of decoder
        sample_y, y_id = self.greedy_cat_connector(py_logits, self.use_gpu, return_max_id=True)

        # pack attention context
        attn_inputs = None
        labels = out_utts[:, 1:].contiguous()

        sample_y = sample_y.view(-1, self.config.k * self.config.y_size)
        dec_init_state = self.c_init_connector(sample_y) + c_last.unsqueeze(0)

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size, out_utts[:, 0:-1], dec_init_state,
                                                   attn_context=attn_inputs,
                                                   mode=TEACH_FORCE, gen_type="greedy",
                                                   beam_size=self.config.beam_size)

        enc_dec_nll = self.nll_loss(dec_outs, labels)
        return Pack(nll=enc_dec_nll)

    def min_forward(self, data_feed, batch_size, sample_n):
        ctx_lens = data_feed['context_lens']
        ctx_utts = self.np2var(data_feed['contexts'], LONG)

        c_inputs = self.utt_encoder(ctx_utts)
        c_outs, c_last = self.ctx_encoder(c_inputs, ctx_lens)
        c_last = c_last.squeeze(0)

        # prior network
        py_logits = self.p_y(F.tanh(self.p_fc1(c_last))).view(-1, self.config.k)
        log_py = F.log_softmax(py_logits, dim=py_logits.dim() - 1)

        temp = []
        temp_ids = []
        # sample the prior network N times
        for i in range(sample_n):
            temp_y, temp_id = self.cat_connector(py_logits, 1.0, self.use_gpu,
                                                 hard=True, return_max_id=True)
            temp_ids.append(temp_id.view(-1, self.config.y_size))
            temp.append(temp_y.view(-1, self.config.k * self.config.y_size))

        sample_y = torch.cat(temp, dim=0)
        y_id = torch.cat(temp_ids, dim=0)
        batch_size *= sample_n
        c_last = c_last.repeat(sample_n, 1)

        # map sample to initial state of decoder
        sample_y = sample_y.view(-1, self.config.k * self.config.y_size)
        dec_init_state = self.c_init_connector(sample_y) + c_last.unsqueeze(0)

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size, None, dec_init_state,
                                                   attn_context=None,
                                                   mode=GEN, gen_type="sample",
                                                   beam_size=self.config.beam_size)
        dec_ctx[DecoderRNN.KEY_LATENT] = y_id
        dec_ctx[DecoderRNN.KEY_POLICY] = log_py
        return dec_ctx

    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)

