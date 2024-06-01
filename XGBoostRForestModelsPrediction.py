import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Chargement des données
df_og = read_csv("stocks/JNJ.csv")
prediction_target = df_og[(df_og['Date'] >= '2019-01-01') & (df_og['Date'] < '2020-01-01')]
df = df_og[(df_og['Date'] < '2019-01-01') & (df_og['Date'] >= '2016-01-01')]
date = df['Date']
df = df.drop(columns=['Date'])

# Hyperparamètres
window_size = 30

# Fonction pour créer une fenêtre glissante
def create_sliding_window(df, window_size):
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df.iloc[i:i + window_size]['Close'])
        y.append(df.iloc[i + window_size]['Close'])
    return np.array(X), np.array(y)

X, y = create_sliding_window(df, window_size)

split_index = int(len(X) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Modèle RandomForest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"RandomForest - Mean Squared Error: {mse_rf}")
print(f"RandomForest - Mean Absolute Error: {mae_rf}")
print(f"RandomForest - RMSE: {np.sqrt(mse_rf)}")

# Modèle XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"XGBoost - Mean Squared Error: {mse_xgb}")
print(f"XGBoost - Mean Absolute Error: {mae_xgb}")
print(f"XGBoost - RMSE: {np.sqrt(mse_xgb)}")

# Refit des modèles sur l'ensemble des données
rf_model.fit(X, y)
xgb_model.fit(X, y)

# Fonction de prévision
def forecast(model, data, window_size, nb_days):
    predictions = []
    current_window = data[-window_size:].tolist()
    for i in range(nb_days):
        prediction = model.predict([current_window])[0]
        predictions.append(prediction)
        current_window.append(prediction)
        current_window.pop(0)
    return predictions

# Prédictions futures
future_predictions_rf = forecast(rf_model, df['Close'].values, window_size, 30)
future_predictions_xgb = forecast(xgb_model, df['Close'].values, window_size, 30)

print("Future predictions RandomForest: ", future_predictions_rf)
print("Future predictions XGBoost: ", future_predictions_xgb)

# Tracé des résultats
plt.plot(prediction_target['Close'][:30].tolist(), label='Prediction Target')
plt.plot(future_predictions_rf, label='RandomForest Prediction', color='green')
plt.plot(future_predictions_xgb, label='XGBoost Prediction', color='orange')

plt.title('Prediction and Reality on Closing Price with 30 Days Window (JNJ)')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
