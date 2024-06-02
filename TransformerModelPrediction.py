import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Chargement des données
df_og = read_csv("stocks/GOOG.csv") #à adapter selon dataset utilisé
#Prediction target temporelle entre 2019 et 2020
prediction_target = df_og[(df_og['Date'] >= '2019-01-01') & (df_og['Date'] < '2020-01-01')]
#Periode d'entraînement du modèle
df = df_og[(df_og['Date'] < '2019-01-01') & (df_og['Date'] >= '2017-01-01')]
date = df['Date']  #Récupération de la colonne des dates
df = df.drop(columns=['Date'])   #Supression de la colonne date dans le dataset

# Normalisation des données
scaler = MinMaxScaler(feature_range=(0, 1))
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Fonction pour créer une fenêtre glissante
def create_sliding_window(df, window_size):
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df.iloc[i:i + window_size]['Close'])
        y.append(df.iloc[i + window_size]['Close'])
    return np.array(X), np.array(y)

#Positional encoding
class PositionalEncoding(nn.Module):
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
        x = x + self.pe[:x.size(0), :]
        return x

# Création du modèle transformer
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, num_heads, num_encoder_layers, dim_feedforward, dropout, max_len=5000):
        super(TransformerTimeSeries, self).__init__()
        self.model_type = 'Transformer'
        self.input_dim = input_dim
        self.positional_encoding = PositionalEncoding(input_dim, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads,
                                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, src):
        src = self.positional_encoding(src.transpose(0, 1))  # Apply positional encoding and transpose for Transformer
        output = self.transformer_encoder(src)
        output = self.fc(output[-1, :, :])
        return output


# Prévision future
def forecast_transformer(model, data, window_size, nb_days):
    model.eval()
    predictions = []
    current_window = torch.tensor(data[-window_size:], dtype=torch.float32).unsqueeze(-1).repeat(1, input_dim)
    for _ in range(nb_days):
        with torch.no_grad():
            src = current_window.unsqueeze(1)
            prediction = model(src).squeeze().mean().item()
        predictions.append(prediction)
        new_window = current_window[1:].clone()
        new_window = torch.cat((new_window, torch.tensor([[prediction] * input_dim])), dim=0)
        current_window = new_window
    return predictions


#Sliding window
#Taille de la fenêtre à adapter selon les tests, parmi la liste
window_size =14  #[73, 30, 20, 14, 7]
prediction_length= 14
X, y = create_sliding_window(df, window_size)

#Séparation des données d'entrainement et de test
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Convertir en tenseurs PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


input_dim = 16  # Doit être divisible par num_heads
num_heads = 8
num_encoder_layers = 3
dim_feedforward = 128
dropout = 0.1

#Instancie le model Transformer
model = TransformerTimeSeries(input_dim, num_heads, num_encoder_layers, dim_feedforward, dropout)

# Ajuster les dimensions d'entrée
X_train = X_train.unsqueeze(-1).repeat(1, 1, input_dim)
X_test = X_test.unsqueeze(-1).repeat(1, 1, input_dim)

# Entraînement du modèle
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

#boucle entrainement du model pour le nombre de boucle choisies
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prédiction sur les données de test
model.eval()
with torch.no_grad():
    test_pred = model(X_test).squeeze()
    test_pred = scaler.inverse_transform(test_pred.numpy().reshape(-1, 1)).squeeze()
    y_test_actual = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).squeeze()

# Calcul des métriques d'erreur
mse_test = mean_squared_error(y_test_actual, test_pred)
mae_test = mean_absolute_error(y_test_actual, test_pred)
print(f"Transformer - Mean Squared Error: {mse_test}")
print(f"Transformer - Mean Absolute Error: {mae_test}")
print(f"Transformer - RMSE: {np.sqrt(mse_test)}")

#Prevision future des prix de fermeture
future_predictions_transformer = forecast_transformer(model, df['Close'].values, window_size, prediction_length)
#Destandardisation des données
future_predictions_transformer = scaler.inverse_transform(
    np.array(future_predictions_transformer).reshape(-1, 1)).squeeze()

print("Future predictions Transformer: ", future_predictions_transformer)

# Tracé des résultats
plt.plot(prediction_target['Close'][:prediction_length].tolist(), label='Prediction Target')
plt.plot(future_predictions_transformer, label='Transformer Prediction', color='blue')

plt.title(f'Prediction and Reality on Closing Price with {window_size} Days Window ')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.legend()

plt.show()