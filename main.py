# coding=utf8
import torch.nn as nn
from torch import optim
from data_util import load_data
from model import SlotRNN

embedding_size = 100
batch_size = 10
n_epochs = 30
learning_rate = 0.01

def eval():
    pass

def train():
    dataset = load_data()

    model = SlotRNN(dataset.vocab_size, embedding_size, dataset.n_classes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    print (model)

    for epoch in range(n_epochs):
        # get batch data
        print_loss = 0
        for data_x, data_y in dataset:
            # zero_grad
            optimizer.zero_grad()
            #forward
            output = model(data_x)
            # compute loss
            loss = criterion(output, data_y)
            print_loss += loss.data[0]
            # backward
            loss.backward()
            optimizer.step()
        print ('epoch: (%d / %d) loss: %.4f' % (epoch+1, n_epochs, print_loss/len(dataset)))




if __name__ == '__main__':
    train()
