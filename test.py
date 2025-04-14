import math
import os
import mplfinance as mpf
import requests
import pandas as pd
import numpy as np


def fetch_data():
    store = "data/eth-210704.csv"
    if os.path.isfile(store):
        # sss
        data1 = pd.read_csv(store, index_col=0, parse_dates=True)
        # data1.shape
        # data1['Date'] = pd.to_datetime(data1['Date'], format="%Y-%m-%d")
        # data1.set_index("Date", inplace=True)
        return data1
    else:
        host = "https://api3.binance.com";
        params = {"symbol": "ETHUSDT", "interval": "1d", "limit": 500}
        response = requests.get(host + "/api/v3/klines", params)

        text = response.json()
        data = pd.DataFrame.from_dict(data=text, orient="columns")
        data.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "close_time", "volumes", "orders", "buy",
                        "sell", "other"]

        data['Date'] = pd.to_datetime(data['Date'], unit="ms")
        data['Open'] = pd.to_numeric(data['Open'])
        data['High'] = pd.to_numeric(data['High'])
        data['Low'] = pd.to_numeric(data['Low'])
        data['Close'] = pd.to_numeric(data['Close'])
        data['Volume'] = pd.to_numeric(data['Volume'])
        data.set_index("Date", inplace=True)

        data.to_csv(store, index=True)

        return data


def show(df, buy, sell):

    df['buy'] = buy
    df['sell'] = sell

    print(df.head(3))

    # apd = [
    #     mpf.make_addplot(df['buy'], type="scatter", scatter=True, markersize=50, marker='^'),
    #     mpf.make_addplot(df['sell'], type="scatter", scatter=True, markersize=50, marker='v'),
    # ]
    mpf.plot(df, type="candle", volume=True, style="binance")
    mpf.show()
    # plot.show()


def label(price):

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


def convert_to_label(buy, sell):

    _y = []
    for i in range(len(buy)):
        if not math.isnan(buy[i]):
            _y.append(1)
        elif not math.isnan(sell[i]):
            _y.append(2)
        else:
            _y.append(0)

    return _y


def slope(seq):
    pos = len(seq)
    if pos > 2:
        seq[pos]-seq[pos-1] > 0


if __name__ == '__main__':

    x = fetch_data()
    # signal_buy, signal_sell = label(x)
    # #print(signal_buy)
    # #print(signal_sell)
    # # print(convert_to_label(signal_buy, signal_sell))
    # show(x, signal_buy, signal_sell)
