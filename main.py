# coding=utf8
import torch
import torch.nn as nn
from torch import optim

from data_util import load_data
from model import SlotRNN
from evaluate import conlleval

embedding_size = 100
n_epochs = 10
learning_rate = 0.01

def train():
    train_dataset, val_dataset = load_data()

    model = SlotRNN(train_dataset.vocab_size, embedding_size, train_dataset.n_classes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    print (model)

    def var2np(variable):
        return torch.max(variable, 1)[1].data.squeeze(1).numpy()

    for epoch in range(n_epochs):
        # get batch data
        print_loss = 0
        train_pred_label = []
        for data_x, data_y in train_dataset:
            # zero_grad
            optimizer.zero_grad()
            #forward
            pred = model(data_x)
            train_pred_label.append(var2np(pred))
            # compute loss
            loss = criterion(pred, data_y)
            print_loss += loss.data[0]
            # backward
            loss.backward()
            optimizer.step()

        train_pred = [list(map(lambda x: train_dataset.idx2labels[x], y)) for y in train_pred_label]
        print conlleval(train_pred, train_dataset.groundtruth, train_dataset.words, 'r.txt')
        print ('epoch: (%d / %d) loss: %.4f' % (epoch+1, n_epochs, print_loss/len(train_dataset)))



if __name__ == '__main__':
    train()
