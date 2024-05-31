import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('stocks/JNJ.csv')
data = df['Close'][(df['Date'] < '2019-01-01') & (df['Date'] >= '2016-01-01')].values

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 30
X, y = create_sequences(data, seq_length)

split_index = int(len(X) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(torch.nn.Module):
    def __init__(self, feature_size=128, num_layers=9, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_size, nhead=16, dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = torch.nn.Linear(feature_size, 1)
        self.feature_size = feature_size

    def forward(self, src):
        src = self.pos_encoder(src.permute(1,0,2))
        output = self.transformer_encoder(src)
        output = self.decoder(output.permute(1,0,2))
        return output[:,-1,:]


feature_size = 64
num_layers = 3
dropout = 0.1

model = TimeSeriesTransformer(feature_size, num_layers, dropout)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 100
batch_size = 64

model.train()
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
# Test
model.eval()
with torch.no_grad():
    test_output = model(X_test)
    test_loss = criterion(test_output.squeeze(), y_test)
    print(f'Test Loss: {test_loss.item()}')

# Prediction
predictions = test_output.squeeze().detach().numpy()

# De-standarisation
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actuals = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
print(predictions)
print(actuals)
plt.plot(actuals, label='Prediction Target',color='blue')
plt.plot(predictions, label='Transformer Prediction',color='black')
plt.title('Transformer model prediction and reality on Closing Price')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

