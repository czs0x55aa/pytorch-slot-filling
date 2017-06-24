# coding=utf8
import gzip
import pickle
import numpy as np
import torch.utils.data as Data

data_path = './data/atis.fold1.pkl.gz'

def load_data():
    f = gzip.open(data_path, 'rb')

    train_set, valid_set, test_set, dicts = pickle.load(f)
    # print np.shape(train_set)
    # print np.shape(valid_set)
    # print np.shape(test_set)

    train_x, _, train_label = train_set
    val_x, _, val_label = valid_set
    # Create index to word/label dicts
    w2idx, labels2idx = dicts['words2idx'], dicts['labels2idx']
    idx2w = {w2idx[k]: k for k in w2idx}
    idx2labels = {labels2idx[k]: k for k in labels2idx}

    n_vocab = len(idx2w)
    n_classes = len(idx2labels)


if __name__ == '__main__':
    load_data()
