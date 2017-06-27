# coding=utf8
import gzip
import pickle
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable

data_path = './data/atis.fold1.pkl.gz'

def load_data():
    f = gzip.open(data_path, 'rb')

    train_set, valid_set, test_set, dicts = pickle.load(f)
    # print np.shape(train_set)
    # print np.shape(valid_set)
    # print np.shape(test_set)

    train_x, _, train_label = train_set
    val_x, _, val_label = valid_set
    test_x, _, test_label = test_set

    # Create index to word/label dicts
    w2idx, labels2idx = dicts['words2idx'], dicts['labels2idx']
    idx2w = {w2idx[k]: k for k in w2idx}
    idx2labels = {labels2idx[k]: k for k in labels2idx}

    train_dataset = DataSet(train_x, train_label, idx2w, idx2labels)
    val_dataset = DataSet(val_x, val_label, idx2w, idx2labels)
    return train_dataset, val_dataset

class DataSet(object):
    def __init__(self, src, target, idx2w, idx2labels, batch_size=1, shuffle=False):
        # assert np.shpae(data)[0] == np.shape(target)[0]
        self.data = zip(src, target)
        # if shuffle is True:
        #     pass
        # cannot support batch
        self.batch_size = batch_size
        self.length = np.shape(src)[0]

        self.words = [list(map(lambda x: idx2w[x], w)) for w in src]
        self.groundtruth = [list(map(lambda x: idx2labels[x], w)) for w in target]

        self.idx2w = idx2w
        self.idx2labels = idx2labels
        self.vocab_size = len(idx2w)
        self.n_classes = len(idx2labels)

    def __getitem__(self, index):
        # assert index < self.length
        pairs = self.data[index]

        def wrap(pairs):
            # pairs.sort(key=lambda x: len(x[0]), reverse=True)
            # data_x, data_y = zip(*pairs)
            data_x, data_y = pairs
            x_variable = Variable(torch.from_numpy(data_x).long())
            y_variable = Variable(torch.from_numpy(data_y).long())
            return x_variable, y_variable
        return wrap(pairs)

    def __len__(self):
        return self.length
