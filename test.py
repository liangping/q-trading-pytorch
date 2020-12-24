import os

import matplotlib.pyplot as plt
import mplfinance as mpf
import requests
import pandas as pd
import numpy as np


def fetch_data():
    store = "data/btc.csv"
    if os.path.isfile(store):
        # sss
        data1 = pd.read_csv(store, index_col=0, parse_dates=True)
        data1.shape
        # data1['Date'] = pd.to_datetime(data1['Date'], format="%Y-%m-%d")
        # data1.set_index("Date", inplace=True)
        return data1
    else:
        host = "https://api3.binance.com";
        params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 500}
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


def show(df):

    buy, sell = label(df)
    df['buy'] = buy
    df['sell'] = sell

    print(df.head(3))

    apd = [
        mpf.make_addplot(df['buy'], type="scatter", scatter=True, markersize=20, marker='^'),
        mpf.make_addplot(df['sell'], type="scatter", scatter=True, markersize=20),
    ]
    mpf.plot(df, type="candle", volume=True, addplot=apd, style="sas")
    mpf.show()
    # plot.show()


def label(price):

    # buy = pd.DataFrame(columns=('Date', 'other'))
    buy = []
    sell = []

    highest = 0
    lowest = 0
    index = 0
    for date, p in price['Close'].iteritems():
        index += 1
        if index % 5 == 0:
            buy.append(p * 0.98)
        else:
            buy.append(np.nan)

        if index % 7 == 0:
            sell.append(p * 1.02)
        else:
            sell.append(np.nan)

    return buy, sell


if __name__ == '__main__':

    x = fetch_data()
    show(x)
    # l_b, l_s = lablel(x)
