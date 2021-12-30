import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextRNN(nn.Module):
    def __init__(self, args, text_emb):
        super(TextRNN, self).__init__()
        self.embedding = text_emb
        # self.embedding = nn.Embedding(args.vocab_size, args.d_model // 1, padding_idx=0)
        self.gru = nn.LSTM(args.pretrain_dim, args.pretrain_dim, num_layers=1,
                           bidirectional=False, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(args.d_model // 2, args.vocab_size)

    def forward(self, inputs, length=None, hidden=None, task='main'):

        embed_inputs = self.embedding(inputs)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths, batch_first=True)

        self.gru.flatten_parameters()
        outputs, hidden = self.gru(embed_inputs, hidden)

        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]
        if task == 'LM':
            outputs = self.fc(outputs)
        return outputs, hidden
