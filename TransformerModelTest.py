import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


# Chargement des données
df = pd.read_csv('stocks/JNJ.csv') #à adapter selon dataset utilisé
#Prediction target temporelle entre 2019 et 2013
data = df['Close'][(df['Date'] < '2019-01-01') & (df['Date'] >= '2013-01-01')].values

# Normalisation des données
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

# Fonction pour créer les séquences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Création des séquences
seq_length = 14 #a adapter selon efficacité
X, y = create_sequences(data, seq_length)

#Séparation des données d'entrainement et de test
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Conversion en tenseurs PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Assurer que la cible a la forme (N, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)    # Assurer que la cible a la forme (N, 1)

#Etape du positional encoding
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

    # Création du cycle entre les encoders et decoders
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Création du modèle transformer
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
        src = self.pos_encoder(src.permute(1, 0, 2))
        output = self.transformer_encoder(src)
        output = self.decoder(output.permute(1, 0, 2))
        return output[:, -1, :].view(-1, 1)  # Assurer que la sortie a la forme (N, 1)

#Initie les hyperparamètres
feature_size = 64
num_layers = 3
dropout = 0.1

#Instancie le model Transformer
model = TimeSeriesTransformer(feature_size, num_layers, dropout)

criterion = torch.nn.MSELoss() #fonction de perte
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  #Optimiser Adam avec learning rate de 0.001

#boucle entrainement du model pour le nombre de boucle choisies
num_epochs = 100
batch_size = 64
model.train()
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Test et évaluation
model.eval()
with torch.no_grad():
    test_output = model(X_test)
    test_loss = criterion(test_output, y_test)
    print(f'Loss de test: {test_loss.item()}')

# Prédiction
predictions = test_output.squeeze().detach().numpy()

# Dénormalisation
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actuals = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# Derniers 7 jours
predictions_last7 = predictions[-14:]
actuals_last7 = actuals[-14:]

# Calcul des métriques
mse = mean_squared_error(actuals_last7, predictions_last7)
mae = mean_absolute_error(actuals_last7, predictions_last7)
rmse = sqrt(mse)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

# Visualisation
plt.plot(actuals_last7, label='Prix de clôture réel', color='blue')
plt.plot(predictions_last7, label='Prix de clôture prédit', color='black')
plt.title('Prediction and Reality on Closing Price with 20 Days Window')
plt.xlabel('Jours')
plt.ylabel('Prix de clôture')
plt.legend()
plt.show()