from .model_bases import LAED


class LightAeED(LAED):
    def __init__(self, corpus, config):
        super(LightLAED, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        if not hasattr(config, "freeze_step"):
            config.freeze_step = 6000

        # build model here
        # word embeddings
        self.x_embedding = nn.Embedding(self.vocab_size, config.embed_size)

        # latent action learned
        self.x_encoder = EncoderRNN(config.embed_size,
                                    config.dec_cell_size,
                                    dropout_p=config.dropout,
                                    rnn_cell=config.rnn_cell,
                                    variable_lengths=False)

        self.q_y = nn.Linear(config.dec_cell_size, config.y_size * config.k)
        self.x_init_connector = nn_lib.LinearConnector(config.y_size * config.k,
                                                       config.dec_cell_size,
                                                       config.rnn_cell == 'lstm')
        # Encoder-Decoder STARTS here
        self.embedding = nn.Embedding(self.vocab_size,
                                      config.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.utt_encoder = RnnUttEncoder(config.utt_cell_size,
                                         config.dropout,
                                         use_attn=config.utt_type == 'attn_rnn',
                                         vocab_size=self.vocab_size,
                                         embedding=self.embedding)

        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size,
                                      config.ctx_cell_size,
                                      0.0,
                                      config.dropout,
                                      config.num_layer,
                                      config.rnn_cell,
                                      variable_lengths=self.config.fix_batch)
        # FNN to get Y
        self.p_fc1 = nn.Linear(config.ctx_cell_size, config.ctx_cell_size)
        self.p_y = nn.Linear(config.ctx_cell_size, config.y_size * config.k)

        # connector
        self.c_init_connector = nn_lib.LinearConnector(config.y_size * config.k,
                                                       config.dec_cell_size,
                                                       config.rnn_cell == 'lstm')
        self.decoder = nn_lib.LinearConnector(config.dec_cell_size, self.vocab_size)
        # decoder

        # force G(z,c) has z
        if config.use_attribute:
            self.attribute_loss = criterions.NLLEntropy(-100, config)

        self.cat_connector = nn_lib.GumbelConnector()
        self.greedy_cat_connector = nn_lib.GreedyConnector()
        # self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.main_loss = torch.nn.BCEWithLogitsLoss()
        self.cat_kl_loss = criterions.CatKLLoss()
        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        self.entropy_loss = criterions.Entropy()

        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
        self.kl_w = 0.0

    def valid_loss(self, loss, batch_cnt=None):
        vae_loss = loss.vae_nll + loss.reg_kl
        enc_loss = loss.nll
        if self.config.greedy_q:
            enc_loss += loss.pi_nll
        else:
            enc_loss += loss.pi_kl

        if self.config.use_attribute:
            enc_loss += 0.1*loss.attribute_nll

        if batch_cnt is not None and batch_cnt > self.config.freeze_step:
            total_loss = enc_loss
            if self.kl_w == 0.0:
                self.kl_w = 1.0
                self.flush_valid = True
                for param in self.x_embedding.parameters():
                    param.requires_grad = False
                for param in self.x_encoder.parameters():
                    param.requires_grad = False
                for param in self.q_y.parameters():
                    param.requires_grad = False
                for param in self.x_init_connector.parameters():
                    param.requires_grad = False
                for param in self.x_decoder.parameters():
                    param.requires_grad = False
        else:
            total_loss = vae_loss

        return total_loss

    def pxz_forward(self, batch_size, results, out_utts, mode, gen_type):
        # map sample to initial state of decoder
        dec_init_state = self.x_init_connector(results.sample_y)
        dec_outs, dec_last, dec_ctx = self.x_decoder(batch_size,
                                                     out_utts[:, 0:-1],
                                                     dec_init_state,
                                                     mode=mode,
                                                     gen_type=gen_type,
                                                     beam_size=self.config.beam_size)
        results['dec_outs'] = dec_outs
        results['dec_ctx'] = dec_ctx

        return results

    def forward(self, data_feed, mode, sample_n=1, gen_type='greedy', return_latent=False):
        ctx_lens = data_feed['context_lens']
        batch_size = len(ctx_lens)

        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # First do VAE here
        vae_resp = self.pxz_forward(batch_size,
                                    self.qzx_forward(out_utts[:, 1:]),
                                    out_utts,
                                    mode,
                                    gen_type)

        # context encoder
        c_inputs = self.utt_encoder(ctx_utts)
        c_outs, c_last = self.ctx_encoder(c_inputs, ctx_lens)
        c_last = c_last.squeeze(0)

        # prior network
        py_logits = self.p_y(F.tanh(self.p_fc1(c_last))).view(-1, self.config.k)
        log_py = F.log_softmax(py_logits, dim=py_logits.dim()-1)

        if mode != GEN:
            sample_y, y_id = vae_resp.sample_y.detach(), vae_resp.y_ids.detach()
            y_id = y_id.view(-1, self.config.y_size)
            qy_id = y_id
        else:
            qy_id = vae_resp.y_ids.detach()
            qy_id = qy_id.view(-1, self.config.y_size)

            if sample_n > 1:
                if gen_type == 'greedy':
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

                elif gen_type == 'sample':
                    sample_y, y_id = self.greedy_cat_connector(py_logits, self.use_gpu, return_max_id=True)
                    sample_y = sample_y.view(-1, self.config.k*self.config.y_size).repeat(sample_n, 1)
                    y_id = y_id.view(-1, self.config.y_size).repeat(sample_n, 1)
                    c_last = c_last.repeat(sample_n, 1)
                    batch_size *= sample_n

                else:
                    raise ValueError
            else:
                sample_y, y_id = self.cat_connector(py_logits, 1.0, self.use_gpu, hard=True, return_max_id=True)

        # pack attention context
        if self.config.use_attn:
            attn_inputs = c_outs
        else:
            attn_inputs = None

        # map sample to initial state of decoder
        sample_y = sample_y.view(-1, self.config.k * self.config.y_size)
        dec_init_state = self.c_init_connector(sample_y) + c_last.unsqueeze(0)

        outputs = self.decoder(dec_init_state)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_ctx[DecoderRNN.KEY_RECOG_LATENT] = qy_id
        dec_ctx[DecoderRNN.KEY_LATENT] = y_id
        dec_ctx[DecoderRNN.KEY_POLICY] = log_py

        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # VAE-related Losses
            log_qy = F.log_softmax(vae_resp.qy_logits, dim=1)
            vae_nll = self.main_loss(vae_resp.dec_outs, labels)
            avg_log_qy = torch.exp(log_qy.view(-1, self.config.y_size, self.config.k))
            avg_log_qy = torch.log(torch.mean(avg_log_qy, dim=0) + 1e-15)
            reg_kl = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)

            # Encoder-decoder Losses
            enc_dec_nll = self.nll_loss(outputs, labels)
            pi_kl = self.cat_kl_loss(log_qy.detach(), log_py, batch_size, unit_average=True)
            pi_nll = F.cross_entropy(py_logits.view(-1, self.config.k), y_id.view(-1))
            _, max_pi = torch.max(py_logits.view(-1, self.config.k), dim=1)
            pi_err = torch.mean((max_pi != y_id.view(-1)).float())

            if self.config.use_attribute:
                pad_embeded = self.x_embedding.weight[self.rev_vocab[PAD]].view(1, 1, self.config.embed_size)
                pad_embeded = pad_embeded.repeat(batch_size, dec_outs.size(1), 1)
                mask = torch.sign(labels).float().unsqueeze(2)
                dec_out_p = torch.exp(dec_outs.view(-1, self.vocab_size))
                dec_out_embedded = torch.matmul(dec_out_p, self.x_embedding.weight)
                dec_out_embedded = dec_out_embedded.view(-1, dec_outs.size(1), self.config.embed_size)
                valid_out_embedded = mask * dec_out_embedded + (1.0-mask) * pad_embeded

                x_outs, x_last = self.x_encoder(valid_out_embedded)
                x_last = x_last.transpose(0, 1).contiguous().view(-1, self.config.dec_cell_size)
                qy_logits = self.q_y(x_last).view(-1, self.config.k)
                attribute_nll = F.cross_entropy(qy_logits, y_id.view(-1).detach())

                _, max_qy = torch.max(qy_logits.view(-1, self.config.k), dim=1)
                adv_err = torch.mean((max_qy != y_id.view(-1)).float())
            else:
                attribute_nll = None
                adv_err = None

            results = Pack(nll=enc_dec_nll,
                           pi_kl=pi_kl,
                           attribute_nll=attribute_nll,
                           vae_nll=vae_nll,
                           reg_kl=reg_kl,
                           mi=mi,
                           pi_nll=pi_nll,
                           pi_err=pi_err,
                           adv_err=adv_err)

            if return_latent:
                results['log_py'] = log_py
                results['log_qy'] = log_qy
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = y_id

            return results