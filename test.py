import os

import matplotlib.pyplot as plt
import mplfinance as mpf
import requests
import pandas as pd


def fetch_data():
    store = "data/btc.csv"
    if os.path.isfile(store):
        # sss
        data1 = pd.read_csv(store)
        data1['Date'] = pd.to_datetime(data1['Date'], format="%Y-%m-%d")
        data1.set_index("Date", inplace=True)
        return data1
    else:
        host = "https://api3.binance.com";
        params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 500}
        response = requests.get(host + "/api/v3/klines", params)

        text = response.json()
        data = pd.DataFrame.from_dict(data=text, orient="columns")
        data.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "close_time", "volumes", "orders", "buy", "buy",
                        "other"]

        data['Date'] = pd.to_datetime(data['Date'], unit="ms")
        data['Open'] = pd.to_numeric(data['Open'])
        data['High'] = pd.to_numeric(data['High'])
        data['Low'] = pd.to_numeric(data['Low'])
        data['Close'] = pd.to_numeric(data['Close'])
        data['Volume'] = pd.to_numeric(data['Volume'])
        data.set_index("Date", inplace=True)

        data.to_csv(store, index=True)

        return data


def show(data):

    print(data.head(3))

    fig, ax = plt.subplots(figsize=(1200/72, 480/72))
    fig.subplots_adjust(bottom=0.1)
    apdict = mpf.make_addplot(data['Low'])
    #mpf.plot(data, type="candle", volume=True, style="binance")
    mpf.plot(data, volume=True, addplot=apdict, style="binance")
    #mpf.plot(data, type="renko")
    ax.grid(True)
    ax.xaxis_date()
    plt.show()


def lablel(data):

    print(len(data))
    highest = 0
    h_i = 0
    lowest = 0
    l_i = 0
    for i in range(len(data)):
        row = data.iloc[i]
        if highest < row["High"]:
            highest = row["High"]
            h_i = i
        if h_i+1 == i:
            lowest = row["Low"]
            print(i, " High:", highest)
        if lowest > row["Low"]:
            lowest = row["Low"]
            l_i = i
        if l_i+1 == i:
            highest = row["High"]
            print(i, " lowest:", lowest)

    #print("High:", highest, " lowest:", lowest)


def percentB_aboveone(percentB, price):
    import numpy as np
    signal = []
    previous = 2
    for date, value in percentB.iteritems():
        if not value <= 1 and previous <= 1:
            signal.append(price[date]*1.01)
        else:
            signal.append(np.nan)
        previous = value
    return signal


if __name__ == '__main__':

    data = fetch_data()
    show(data)
    #lablel(data)
