import sys
import numpy as np, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

from models_.video_encoder import VisualFront
from models_.modules_.transformer import PositionalEncoding, make_transformer_encoder, make_transformer_decoder, Embeddings
from models_.transducer import Transducer

from utils.utils import subsequent_mask
from utils.utils import WordEncoder as WE, CharEncoder, BytePairEncoder as BPE

import criteria
from itertools import groupby
from warprnnt_pytorch import RNNTLoss
from torch import cosine_similarity
from time import time

sys.path.append('../')


def get_Emask_chunk(m, K, H):
    L = m.size(0)
    for i in range(L // K + 1):
        m[i * K:(i + 1) * K, max(0, i * K - H):(i + 1) * K] = 1
    return m


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        d = args.d_model
        self.visual_front_encoder = VisualFront(args)

        self.embedding = Embeddings(args)
        self.pos_encoder = PositionalEncoding(args.d_model, dropout=args.dropout)
        self.transformer_encoder = make_transformer_encoder(args.num_layers // 2, args.d_model, args.d_ff,
                                                            args.num_heads, args.dropout,
                                                            args.ffn_layer, args.first_kernel_size)
        self.transformer_encoder_mem = make_transformer_encoder(1, args.d_model, args.d_ff,
                                                                args.num_heads, args.dropout,
                                                                args.ffn_layer, args.first_kernel_size)
        self.transformer_decoder = make_transformer_decoder(args.num_layers, args.d_model, args.d_ff,
                                                            args.num_heads, args.dropout,
                                                            'fc', args.first_kernel_size)

        self.rnn_t = Transducer(args, self.embedding)

        self.ctc_linears = nn.Sequential(
            nn.Linear(args.d_model, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, args.vocab_size)
        )
        self.chunk, history = args.chunk_size, 10
        self.mem_size, self.context_size = args.mem_size, 0
        self.enc_mask = get_Emask_chunk(torch.zeros(self.args.video_max_length, self.args.video_max_length), K=self.chunk, H=history).bool()

        self.conv_v_mem = nn.Conv1d(d, d, kernel_size=self.chunk, stride=self.chunk, padding=0, bias=False)
        self.maxpool_v_mem = nn.MaxPool1d(kernel_size=self.chunk, stride=self.chunk, padding=0)
        self.avgpool_v_mem = nn.AvgPool1d(kernel_size=self.chunk, stride=self.chunk, padding=0)

        self.alpha = 0.2

    def forward(self, x, x_mask, y=None, y_mask=None, **batch):
        ctc_loss = rnnt_loss = rnnt_pred = ctc_pred = ALs = torch.Tensor([0]).to(x.device)
        bs = x.size(0)
        x_len = torch.sum(x_mask, dim=1).int()  # [bs]
        x = x[:, :max(x_len)]
        x_mask = x_mask[:, :max(x_len)]
        if batch['train']:
            y_len = torch.sum(y_mask, dim=1).int()
            y = y[:, :max(y_len)]
            y_mask = y_mask[:, :max(y_len)]

        x = self.visual_front_encoder(x.unsqueeze(1))
        x_mask_ = self.enc_mask[:max(x_len), :max(x_len)].to(x.device) & x_mask.unsqueeze(-2)

        tf_enc = self.transformer_encoder(self.pos_encoder(x), x_mask_)
        v_enc = tf_enc

        if self.mem_size > 0:
            update = True
            x = tf_enc.detach()
            tf_enc_m = torch.zeros(x.size()).to(x.device)
            m_score = torch.zeros(bs, 0).to(x.device)
            m_freq = torch.zeros(bs, 0).to(x.device)

            C, M = self.chunk, self.mem_size
            x = self.pos_encoder(x)
            tf_enc_m[:, 0: C] = self.transformer_encoder_mem(x[:, 0: C],
                                                             x_mask[:, 0: C].unsqueeze(-1))

            mem = new_m = self.maxpool_v_mem(x[:, 0: C].permute(0, 2, 1)).permute(0, 2, 1)

            mems = [mem]
            mem_action = []
            mem_seq = []
            for i in range(1, (x.size(1) - 1) // C + 1):
                tf_enc_m[:, i * C:(i + 1) * C], _, score = self.transformer_encoder_mem(x[:, i * C:(i + 1) * C],
                                                                                        x_mask[:, i * C:(i + 1) * C].unsqueeze(-1),
                                                                                        mem)
                if update:
                    new_m = self.maxpool_v_mem(F.pad(x[:, i * C:(i + 1) * C], pad=(0, 0, 0, self.chunk - 1), value=0).permute(0, 2, 1)).permute(0, 2, 1)
                    mem_seq.append([mem.tolist(), new_m.tolist()])
                    entropy = (-torch.log2(score + 1e-10) * score).sum(1).mean()  # ?
                    threshold = np.log2(int(mem.size(1)) + 1e-10) * 0.5  # ?

                    if mem.size(1) > M:
                        if entropy > threshold:
                            m_score = torch.cat([m_score, torch.zeros(bs, new_m.size(1)).to(x.device)], dim=1)
                            m_freq = torch.cat([m_freq + 1, torch.ones(bs, new_m.size(1)).to(x.device)], dim=1)
                            m_score += score

                            topK_sc, topK_idx = torch.topk(m_score / m_freq, k=M)
                            replace_idx = torch.argsort(m_score / m_freq, descending=False)[0, :mem.size(1) - M]
                            topK_idx = torch.sort(topK_idx)[0]

                            m_score_ = torch.zeros(bs, M).to(x.device)
                            m_freq_ = torch.zeros(bs, M).to(x.device)
                            mem_ = torch.zeros(bs, M, mem.size(-1)).to(x.device)
                            for b in range(bs):
                                m_score_[b] = m_score[b, topK_idx[b]]
                                m_freq_[b] = m_freq[b, topK_idx[b]]
                                mem_[b] = mem[b, topK_idx[b]]
                            m_score, m_freq, mem = m_score_, m_freq_, mem_
                            mem = torch.cat([mem, new_m], dim=1)  # [:, max(0, mem.size(1) - self.mem_size):]
                            mem_action.append([i, entropy.item(), 1, replace_idx.item(), ])

                        elif entropy <= threshold:
                            mem_ = torch.zeros(mem.size()).to(x.device)
                            for b in range(bs):
                                for j in range(new_m.size(1)):
                                    sim = cosine_similarity(new_m[b, j].unsqueeze(0), mem[b])
                                    s, idx = torch.max(sim, dim=0)
                                    s = s.item()
                                    mem_[b, idx] = 0.7 * new_m[b, j] - 0.7 * mem[b, idx]
                            mem = mem + mem_

                            mem_action.append([i, entropy.item(), 2, idx.item(), ])

                    elif mem.size(1) < 3 or entropy > threshold:
                        m_score = torch.cat([m_score, torch.zeros(bs, new_m.size(1)).to(x.device)], dim=1)
                        m_freq = torch.cat([m_freq + 1, torch.ones(bs, new_m.size(1)).to(x.device)], dim=1)
                        m_score += score
                        mem = torch.cat([mem, new_m], dim=1)
                    elif entropy <= threshold:
                        mem_ = torch.zeros(mem.size()).to(x.device)
                        for b in range(bs):
                            for j in range(new_m.size(1)):
                                sim = cosine_similarity(new_m[b, j].unsqueeze(0), mem[b])
                                s, idx = torch.max(sim, dim=0)
                                s = s.item()
                                mem_[b, idx] = 0.7 * new_m[b, j] - 0.7 * mem[b, idx]
                        mem = mem + mem_
                        mem_action.append([i, entropy.item(), 2, idx.item(), ])

                mems.append(mem)

            v_enc = v_enc + tf_enc_m * self.alpha

        # ctc_loss and rnnt_loss
        ctc_input = torch.log_softmax(self.ctc_linears(v_enc), dim=-1).transpose(0, 1)  # [v_len, bs, vocab_size]
        if batch['train']:

            ctc_loss = nn.CTCLoss(blank=0, reduction='mean')(ctc_input, y, x_len, y_len)
            if batch['epoch'] > self.args.CTC_epoch and 'rnnt' in self.args.dec:
                y_pad = F.pad(y, pad=(1, 0), value=1)  # <SOS>
                y_mask_pad = F.pad(y_mask, pad=(1, 0), value=True)
                y_mask_pad_ = y_mask_pad.unsqueeze(-2) & subsequent_mask(y_mask_pad.size(-1)).to(y_mask.device)
                rnnt_input = self.rnn_t(v_enc, y_pad, y_len + 1, y_mask_pad_, v_mem=None)  # mem
                rnnt_loss = RNNTLoss(blank=0, reduction='mean')(rnnt_input, y.int().contiguous(), x_len, y_len)
                rnnt_loss = rnnt_loss[0]

        if not batch['train'] and 'rnnt' in self.args.dec:
            rnnt_pred, ALs = self.rnn_t.recognize(v_enc, x_len, y=y)

        if not batch['train'] and 'ctc' in self.args.dec:
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                inputs=ctc_input.cpu().detach().numpy(),
                sequence_length=x_len.cpu().numpy(),
                beam_width=1,
                top_paths=1, )
            ctc_decode = ctc_decode[0]

            tmp_sequences = [[] for _ in range(bs)]
            for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
                tmp_sequences[dense_idx[0]].append(ctc_decode.values[value_idx].numpy() + 0)
            ctc_pred = []
            for seq_idx in range(0, len(tmp_sequences)):
                ctc_pred.append([x[0] for x in groupby(tmp_sequences[seq_idx])])


        return {
            'v_enc': v_enc,
            'v_enc_mask': x_mask,
            'loss': ctc_loss + rnnt_loss,
            'ctc_loss': ctc_loss,
            'rnnt_loss': rnnt_loss,
            'ctc_pred': ctc_pred,
            'rnnt_pred': rnnt_pred,
            'ALs': ALs,
        }

    def evaluate(self, pred, y, **batch):
        if self.args.text_level == 'word':
            text_decoder = WE(3, self.args.vocab)
        elif self.args.text_level == 'char':
            text_decoder = CharEncoder(3, self.args.vocab)
        elif self.args.text_level == 'subword':
            text_decoder = BPE(3, self.args.vocab)
        predict_text = []
        for i in range(len(y)):
            cur = []
            for p in pred[i]:
                if p == text_decoder.reserved_ids['<EOS>']:
                    break
                if p != text_decoder.reserved_ids['<PAD>']:
                    cur.append(p)
            cur_str = text_decoder.decode(cur)
            predict_text.append(cur_str)

        cers, wers = [], []
        for t, p in zip(y, predict_text):
            cer = criteria.get_word_error_rate(t, p)
            wer = criteria.get_word_error_rate(t.split(), p.split())
            cers.append(cer)
            wers.append(wer)
            if self.args.evaluate:
                print(t, '\t', p)
        return np.mean(cers), np.mean(wers)