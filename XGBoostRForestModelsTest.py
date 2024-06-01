import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv('GOOG.csv')
data = df['Close'][(df['Date'] < '2019-01-01') & (df['Date'] >= '2017-01-01')].values

# Normalisation des données
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

# Fonction pour créer des séquences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Création des séquences
seq_length = 73
X, y = create_sequences(data, seq_length)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Conversion en 2D pour les modèles XGBoost et RandomForest
X_train_2d = X_train.reshape(X_train.shape[0], -1)
X_test_2d = X_test.reshape(X_test.shape[0], -1)

# Entraînement du modèle XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train_2d, y_train)

# Prédiction avec XGBoost
xgb_predictions = xgb_model.predict(X_test_2d)

# Entraînement du modèle RandomForest
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train_2d, y_train)

# Prédiction avec RandomForest
rf_predictions = rf_model.predict(X_test_2d)

# Dénormalisation
xgb_predictions = scaler.inverse_transform(xgb_predictions.reshape(-1, 1))
rf_predictions = scaler.inverse_transform(rf_predictions.reshape(-1, 1))
actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

# Derniers 14 jours
xgb_predictions_last14 = xgb_predictions[-30:]
rf_predictions_last14 = rf_predictions[-30:]
actuals_last14 = actuals[-30:]

# Calcul des métriques pour XGBoost
xgb_mse = mean_squared_error(actuals_last14, xgb_predictions_last14)
xgb_mae = mean_absolute_error(actuals_last14, xgb_predictions_last14)
xgb_rmse = sqrt(xgb_mse)

# Calcul des métriques pour RandomForest
rf_mse = mean_squared_error(actuals_last14, rf_predictions_last14)
rf_mae = mean_absolute_error(actuals_last14, rf_predictions_last14)
rf_rmse = sqrt(rf_mse)

print('XGBoost Metrics:')
print(f'MSE: {xgb_mse}')
print(f'MAE: {xgb_mae}')
print(f'RMSE: {xgb_rmse}')

print('RandomForest Metrics:')
print(f'MSE: {rf_mse}')
print(f'MAE: {rf_mae}')
print(f'RMSE: {rf_rmse}')

# Visualisation des résultats
plt.figure(figsize=(12, 6))
plt.plot(actuals_last14, label='Prix de clôture réel', color='blue')
plt.plot(xgb_predictions_last14, label='Prix de clôture prédit (XGBoost)', color='green')
plt.plot(rf_predictions_last14, label='Prix de clôture prédit (RandomForest)', color='red')
plt.title('Prediction and Reality on Closing Price with 73 Days Window')
plt.xlabel('Jours')
plt.ylabel('Prix de clôture')
plt.legend()
plt.show()
