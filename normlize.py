import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import kstest, skew, kurtosis


def normalize(path):
    if not os.path.isfile(path):
        return None
        # sss
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    return data

def rolling_z_score(data, window=30):
    rolling_mean = data.rolling(window=window, min_periods=1).mean()
    rolling_std = data.rolling(window=window, min_periods=1).std()
    return (data - rolling_mean) / rolling_std

def rolling_min_max_scaler(data, window=30):
    rolling_min = data.rolling(window=window, min_periods=1).min()
    rolling_max = data.rolling(window=window, min_periods=1).max()
    return (data - rolling_min) / (rolling_max - rolling_min)

def trend(data):
    print(data)
    data['Open'] = rolling_z_score(data['Open'])
    data['High'] = rolling_z_score(data['High'])
    data['Low'] = rolling_z_score(data['Low'])
    data['Close'] = rolling_z_score(data['Close'])
    data['Volume'] = rolling_z_score(data['Volume'])
    print(data.head(7))
    
    data_show = data.head(30)

    x_data = data_show.index
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, data_show['Close'])
    # plt.plot(x_data, buys, marker='o', markersize=8, markerfacecolor='r')
    # plt.plot(x_data, sells, marker='o', markersize=8, )
    plt.title(path)
    plt.legend(['close'], loc='upper left')
    plt.show()

def distribution(data):

    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    stat, p = kstest(data, 'norm')
    print('Statistics=%.3f, p=%.3f' % (stat, p))


    data_skewness = skew(data)
    data_kurtosis = kurtosis(data)

    print(f'Skewness: {data_skewness}')
    print(f'Kurtosis: {data_kurtosis}')

    plt.figure(figsize=(10, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.show()

if __name__ == '__main__':
    # data = normalize("data/btc-210704.csv") 
    # path = "data/eth-210704.csv" 
    path = "data/btc-210704.csv"
    # path = "data/GSPC.csv"
    # path = "data/HSI.csv"
    data = normalize(path)  
 
    rolled_data = rolling_z_score(data['Open'])

    print(rolled_data[1:].head(7))
    distribution(rolled_data[1:])