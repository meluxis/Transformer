import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df_og = read_csv("stocks/JNJ.csv")
prediction_traget=df_og[(df_og['Date'] >= '2019-01-01')&(df_og['Date']<'2020-01-01')]
df = df_og[(df_og['Date'] < '2019-01-01') & (df_og['Date'] >= '2017-01-01')]
date = df['Date']
df=df.drop(columns=['Date'])

#Hyperparameters
window_size = 73

def create_sliding_window(df, window_size):
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df.iloc[i:i + window_size]['Close'])
        y.append(df.iloc[i + window_size]['Close'])
    return np.array(X), np.array(y)
X, y = create_sliding_window(df,window_size)

split_index = int(len(X) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"RMSE: {np.sqrt(mse)}")

model.fit(X,y)

def forecast(model, data, window_size, nb_days):
    predictions=[]
    current_window=data[-window_size:].tolist()
    for i in range(nb_days):
        prediction=model.predict([current_window])[0]
        predictions.append(prediction)
        current_window.append(prediction)
        current_window.pop(0)
    return predictions

future_predictions = forecast(model, df['Close'].values, window_size, 14)
print("Future predictions ", future_predictions)

plt.plot(prediction_traget['Close'][:14].tolist(), label='Prediction Target')
plt.plot(future_predictions, label=' Our Prediction ')

plt.title('Prediction and Reality on closing price with 73 days window (JNJ)')
plt.xlabel('Days')
plt.ylabel('Closing Price')

plt.legend()
plt.show()