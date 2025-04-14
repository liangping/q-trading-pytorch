import torch
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt


def series_generator():
    x = torch.arange(1, 1000, 0.01)
    return torch.sin(x)


def train_data_generator(seq, k):
    data = list()
    l = len(seq)
    for i in range(l - k - 1):
        x = seq[i:i + k]
        y = seq[i + 1:i + k + 1]
        data.append((x, y))
    return data


class LSTMpred(nn.Module):

    def __init__(self, input_size, hidden_dim):
        super(LSTMpred, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_dim)), Variable(torch.zeros(1, 1, self.hidden_dim))

    def forward(self, x):
        lstmout, self.hidden = self.lstm(x.view(len(x), 1, -1), self.hidden)
        out = self.output(lstmout.view(len(x), -1))
        return out


def train():

    data = series_generator()
    data = train_data_generator(data, 6)

    model = LSTMpred(1, 6)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # training
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(10):
            print("start:", epoch)
            for seq, out in data[:700]:
                seq = Variable(seq)
                out = Variable(out)

                optimizer.zero_grad()
                model.init_hidden()
                modout = model(seq)

                # print(modout, modout.reshape(-1))     print(modout.reshape(-1).shape, out.shape)

                loss = loss_function(modout.reshape(-1), out)
                torch.autograd.set_detect_anomaly(True)
                if epoch == 0:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                optimizer.step()

    # testing
    pred = []
    for seq, y in data[700:]:
        seq = Variable(seq)
        yp = model(seq)[-1].data.numpy()[0]
        pred.append(yp)

    fig = plt.figure()
    plt.plot(y.numpy())
    plt.plot(range(700, 999), pred)
    plt.xlabel("Time(s)")
    plt.ylabel("Value")
    plt.legend("Origin:", "Predict:")
    plt.show()


if __name__ == '__main__':
    train()
