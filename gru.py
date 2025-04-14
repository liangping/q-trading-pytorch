# 引入torch相关模块
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from coindataset import signals, signals_to_labels, CoinsDataset
import pandas as pd
import numpy as np


# 定义GRU模型
class GruModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, hidden_layer):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gruLayer = nn.GRU(in_dim, hidden_dim, hidden_layer, batch_first=True)
        self.fcLayer = nn.Linear(hidden_dim, out_dim)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x, h0):
        out, hn = self.gruLayer(x, h0)
        out = self.fcLayer(out.squeeze(0))
        out = self.sm(out)
        return out, hn

    def init_hidden(self):
        return torch.zeros(2, 1, self.hidden_dim)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, _input, hidden):
        combined = torch.cat((_input, hidden), 1)
        hidden = self.i2h(combined)
        _output = self.i2o(combined)
        _output = self.softmax(_output)
        return _output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


BATCH_SIZE = 5


# def prepare_trainning_date():
#     train_data = []
#     for i in range(len(data) - BATCH_SIZE):
#         train_data.append(data.iloc[i:i+BATCH_SIZE])
#     return np.array(train_data)

def train():
    # 输入维度为5，隐藏层维数为128, 输出维度为3，定义LSTM/GRU层数为2
    gru = GruModel(4, 128, 3, 2)
    # gru = RNN(5, 128, 3)
    loss_set = []

    # 定义损失函数和优化函数
    criterion = nn.MSELoss()
    optimizer = optim.Adam(gru.parameters(), lr=1e-4)

    # 处理输入

    train_loader = DataLoader(CoinsDataset("./data/^GSPC.csv"))

    num_epoches = 10
    for epoch in range(num_epoches):
        for i, (x, y) in enumerate(train_loader):
            if i == 0:
                print(x[0])
            # print(torch.Tensor(y).shape)
            # exit(1)
            hidden = gru.init_hidden()
            # print(hidden.shape)
            optimizer.zero_grad()

            for xi in x[0]:
                # print(xi)
                inputs = Variable(xi.unsqueeze(0), requires_grad=True)
                # forward
                output, hidden = gru(inputs, hidden)

            target = Variable(torch.Tensor(y), requires_grad=True)
            loss = criterion(output, target.unsqueeze(0))
            # update parameters
            if i == 0:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            optimizer.step()

            # print training information
            print_loss = loss.item()
            loss_set.append((epoch, print_loss))

        if epoch % 2 == 0:
            print('Epoch[{}/{}], Loss: {:.5f}'.format(epoch, num_epoches, print_loss))

    gru = gru.eval()

    # 预测结果并比较
    #
    test_loader = DataLoader(CoinsDataset("./data/^GSPC_2011.csv"))
    for i, (x, y) in enumerate(test_loader):
        hidden = gru.init_hidden()
        for xi in x[0]:
            inputs = Variable(xi.unsqueeze(0), requires_grad=True)
            # forward
            output, hidden = gru(inputs, hidden)

        print(torch.Tensor(y).argmax(), output[0].argmax())
    # py = np.array(py).reshape(-1)
    # print(px.shape, py.shape, ry.shape)
    #
    # np.save('prediction',py[-36:])
    # np.save('real',ry[-36:])


torch.set_default_tensor_type(torch.DoubleTensor)
torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    train()
