# coding=utf8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np

class SlotRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_classes, bidirectional=False):
        super(SlotRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.batch_size = 1

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, input):
        input_embedded = self.embedding(input.view(self.batch_size, -1))
        # input embedded size (batch_size, seq_len, input_size)
        rnn_out, rnn_hidden = self.rnn(input_embedded, self.initHidden())
        affine_out = self.linear(rnn_out.view(-1, self.hidden_size))
        return F.log_softmax(affine_out)

    def initHidden(self):
        #  (num_layers, batch, hidden_size)
        init_hidden = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        # if USE_CUDA:
        #     init_hidden = init_hidden.cuda()
        return init_hidden
