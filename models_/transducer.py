import json, six, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_.text_encoder import TextRNN
from time import time
from math import ceil


class Transducer(nn.Module):
    def __init__(self, args, embedding):
        super(Transducer, self).__init__()
        self.args = args
        d = args.d_model
        self.embedding = embedding
        self.text_rnn_encoder = TextRNN(args, embedding)

        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(args.pretrain_dim, d)
        self.fc = nn.Linear(d, args.vocab_size)

    def joint(self, v, t):
        if len(v.size()) == 3 and len(t.size()) == 3:
            v = v.unsqueeze(2)
            t = t.unsqueeze(1)
            v = v.repeat([1, 1, t.size(2), 1])
            t = t.repeat([1, v.size(1), 1, 1])
        elif len(v.size()) == 2 and len(t.size()) == 2:
            pass
        enc = self.fc1(v)
        dec = self.fc2(t)
        joint = self.fc(F.relu(enc + dec))
        log_prob = joint.log_softmax(-1)
        return log_prob

    def forward(self, v_feature, t, t_len):
        # v:[bs, v_len, d]
        t_, hidden = self.text_rnn_encoder(t, t_len)  # [bs, t_len, d_model]
        prob = self.joint(v_feature, t_)
        return prob

    def recognize(self, inputs, inputs_length, y=None):
        batch_size = inputs.size(0)
        C = self.args.chunk_size
        enc_states = inputs
        start_token = torch.ones(1, 1).long().to(inputs.device)

        a = time()
        TIME_dict = {}

        def decode(enc_state, lengths, y=None):
            token_list = [1]
            dec_state, hidden = self.text_rnn_encoder(start_token)
            dyi = []
            AL = 0
            video_text_prob_dist = []
            for i in range(lengths):
                log_prob = self.joint(enc_state[i].view(-1), dec_state.view(-1))
                video_text_prob_dist.append(log_prob.softmax(-1).tolist())
                pred = torch.argmax(log_prob, dim=0).item()
                if pred != 0:
                    dyi.append(i + 1)
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]]).to(inputs.device)
                    dec_state, hidden = self.text_rnn_encoder(token, hidden=hidden)
                if (i + 1) % C == 0:
                    TIME_dict[(ceil(i + 1) / C)] = time() - a

            X = ceil(dyi[-1] / C)
            TIME_dict[X] = time() - a
            tau = X / len(y[0].split())
            Ts = C / 30
            for i, L in enumerate(dyi[1:]):
                seg_n = ceil(L / C)
                AL += (seg_n - tau * i) * Ts  # NCA
            AL = AL / len(dyi) * 1000

            return token_list[1:], AL

        results, ALs = [], []

        for i in range(batch_size):
            decoded_seq, AL = decode(enc_states[i], inputs_length[i], y)
            results.append(decoded_seq)
            ALs.append(AL)

        return results, ALs