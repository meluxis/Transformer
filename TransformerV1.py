#Work In Progress
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('stocks/AEE.csv', parse_dates=['Date'], index_col='Date')
data=df['Adj Close'].values
print(data)
scaler=MinMaxScaler(feature_range=(0,1))
data=scaler.fit_transform(data.reshape(-1,1))
print(data)

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 30  # This can be adjusted
X, y = create_sequences(data, seq_length)

