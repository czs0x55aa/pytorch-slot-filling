# coding=utf8

from model import SlotRNN

batch_size = 10
n_epochs = 30
learning_rate = 0.01

def eval():
    pass

def train():
    model = SlotRNN(vocab_size, hidden_size, n_classes)
    optimizer = optimizer.SGD(model.parameters(), lr=learning_rate)
    print (model)

    for epoch in range(n_epochs):
        pass
        # get batch data

        # zero_grad
        optimizer.zero_grad()
        #forward

        # compute loss

        # backward



if __name__ == '__main__':
    train()
