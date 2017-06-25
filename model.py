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

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn_cel = nn.RNN(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, input):
        input_embedded = self.embedding(input)
        print input_embedded.size()
        exit()
        rnn_out, rnn_hidden = self.rnn_cel(input_embedded, self.initHidden())
        exit()
        affine_out = self.linear(rnn_out)
        return F.softmax(affine_out)

    def initHidden(self):
        #  (num_layers * num_directions, batch, hidden_size)
        init_hidden = Variable(torch.zeros(1, 1, self.hidden_size))
        # if USE_CUDA:
        #     init_hidden = init_hidden.cuda()
        return init_hidden
