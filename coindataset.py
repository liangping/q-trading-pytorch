import math

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

BATCH_SIZE = 3


def normalization(x):
    # Implement sigmoid function
    # return 1/(1 + np.exp(-x))
    Min = np.min(x)
    Max = np.max(x)
    return (x - Min) / (Max - Min)


# Class
class CoinsDataset(Dataset):
    def __init__(self, csv_path):
        self.columns = ["Open", "High", "Close", "Low"]
        self.csv_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # self.csv_data = normalization(self.csv_data[self.columns])
        # self.csv_data = torch.Tensor(self.csv_data.values)
        self.buy, self.sell = signals(self.csv_data)
        self.labels = signals_to_labels(self.buy, self.sell)
        print(self.labels)

    def __len__(self):
        return len(self.labels)-BATCH_SIZE

    def __getitem__(self, idx):
        _x = []
        for i in range(3):
            _x.append([self.csv_data.iloc[idx+i].to_numpy()])
        # print(_x)
        return np.array(_x), self.labels[idx+BATCH_SIZE]

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def signals(price):

    index = 0
    buy, sell, seq, last = [], [], [], []

    for date, p in price.iterrows():
        index += 1
        mean = p['Open':'Close'].mean()
        seq.append(mean)
        buy.append(np.nan)
        sell.append(np.nan)

        if len(seq) > 2:
            # print(seq[index-3], seq[index-2], mean, p['Open':'Close'].std())
            if seq[index-3] - seq[index-2] > 0 and mean - seq[index-2] > 0:
                buy[index-2+1] = last['Low']

            if seq[index-3] - seq[index-2] < 0 and mean - seq[index-2] < 0:
                sell[index-2+1] = last['High']

        last = p

    length = len(buy)
    for i in range(length-1):
        if not math.isnan(buy[i]) and not math.isnan(sell[i+1]):
            buy[i] = np.nan
            if i < length:
                sell[i+1] = np.nan
        if not math.isnan(sell[i]) and not math.isnan(buy[i+1]):
            sell[i] = np.nan
            if i < length:
                buy[i+1] = np.nan

    return buy, sell


def signals_to_labels(buy, sell):
    _y = []
    for i in range(len(buy)):
        if not math.isnan(buy[i]):
            _y.append([0, 1, 0])
            # _y.append([1])
        elif not math.isnan(sell[i]):
            _y.append([0, 0, 1])
            # _y.append([2])
        else:
            _y.append([1, 0, 0])
            # _y.append([0])

    return _y


torch.set_default_tensor_type(torch.FloatTensor)

if __name__ == '__main__':

    ds = CoinsDataset("./data/^GSPC.csv")
    ds.__len__()
    print(rolling_window(ds.csv_data, 30))
    print(ds.csv_data)
    print(ds.labels)

    # for i, data in enumerate(ds):
    #
    #     item1, item2 = data
    #     print("Data1:", item1, "====")
    #     print("Data2", item2, "----")
