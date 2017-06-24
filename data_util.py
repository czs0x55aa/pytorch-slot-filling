# coding=utf8
import gzip
import pickle
import numpy as np

data_path = './data/atis.fold1.pkl.gz'

if __name__ == '__main__':
    f = gzip.open(data_path, 'rb')

    train_set, vaild_set, test_set, dicts = pickle.load(f, encoding='latin1')

    print np.shape(train_set)
    print np.shape(vaild_set)
    print np.shape(test_set)
