# coding=utf8
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
    print (model)

    for epoch in range(n_epochs):
        # get batch data
        for data_x, data_y in dataset:
            # zero_grad
            optimizer.zero_grad()
            #forward
            model(data_x)
            exit()
            # compute loss

            # backward



if __name__ == '__main__':
    train()
