from torch.utils.data import Dataset
import pandas as pd

# Class
class CoinsDataset(Dataset):
    def __init__(self, csv_path):
        self.csv_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.buy, self.sell = signals(self.csv_data)
        self.labels = signals_to_labels(self.buy, self.sell)

    def __len__(self):
        return len(self.csv_data)

    def __gettime__(self,idx):
        data = (self.csv_data[idx], self.labels[idx])
        return data


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
            print(seq[index-3], seq[index-2], mean, p['Open':'Close'].std())
            if seq[index-3] - seq[index-2] > 0 and mean - seq[index-2] > 0:
                buy[index-2] = last['Low']

            if seq[index-3] - seq[index-2] < 0 and mean - seq[index-2] < 0:
                sell[index-2] = last['High']

        last = p

    length = len(buy)
    for i in range(length):
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
            _y.append(1)
        elif not math.isnan(sell[i]):
            _y.append(2)
        else:
            _y.append(0)

    return _y


if __name__ == '__main__':

    ds = CoinsDataset("./data/btc.csv")
    print(ds.labels)
