# coding=utf8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np

class SlotRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, out_size, bidirectional=False):
        super(SlotRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.embedding = nn.Embedding(vocab_size, hidden_size, dropout=0.8)
        self.rnn_cel = nn.RNN(hidden_size, hidden_size, bidirectional)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, input):
        input_embedded = self.embedding(input)
        rnn_out, rnn_hidden = self.rnn_cel(input_embedded)
        affine_out = self.linear(rnn_out)
        return F.softmax(affine_out)
