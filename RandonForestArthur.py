# forecast monthly births with random forest
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler

df = read_csv("AEE.csv")
df = df[(df['Date'] <= '2006-03-03') & (df['Date'] >= '2004-03-03')]
date = df['Date']
X = df.drop(columns=['Close', 'Date']).values
y = df['Close'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

RFClassifier = RandomForestRegressor(n_estimators=100, random_state=42)
RFClassifier.fit(X_scaled, y)

last_data_point = X_scaled[-1].reshape(1, -1)
scaled_last_data = scaler.transform(last_data_point)
predicted_last_data = RFClassifier.predict(scaled_last_data)
print(f'Prédiction pour le 3 mai 2006 : {predicted_last_data[0]}')

true_value = [y[-1]]
mae = mean_absolute_error(true_value, predicted_last_data)
mse = mean_squared_error(true_value, predicted_last_data)

print(f'MAE : {mae}')
print(f'MSE : {mse}')

print(f'Actual value: {y[-1]}')
