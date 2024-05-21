import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------
#Code only adaptable for stock market yet
#---------------------------------------------------------------------------------------------
#Date,Open,High,Low,Close,Adj Close,Volume
#50.099998474121094,51.040000915527344,50.040000915527344,50.61000061035156,26.6218318939209,1109100

# Load the dataset
directory = './stocks/AEE.csv'
df = pd.read_csv(directory)
print(df.head())

# Convert 'Date' column to datetime type and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

#define the prediction date
prediction_date = pd.to_datetime('2020-04-01')

#Define the features and target
features = ['Open', 'Low', 'High', 'Close', 'Volume']
target = 'Adj Close'

#list for 2, 4,8.. days prior the prediction date
list_testing_days_prior = [2, 4, 8, 14, 31, 91, 182, 365]

#function to get the features for the prediction date
#for corresponding training set
def get_features_for_prediction_date(data, prediction_date, days_prior):
    start_date = prediction_date - pd.Timedelta(days=days_prior)
    end_date = prediction_date - pd.Timedelta(days=1)
    return data.loc[start_date:end_date, features]

# Get the training data excluding the prediction date
train = df[df.index < prediction_date]

# Extract features and target for training
X_train = train[features]
y_train = train[target]

#apply randomforest model
model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

list_prediction = []
list_error_prediction = []

#Function for all the predictions needed and their corresponding rmse error
def predictions(data, list_testing_days_prior):
    yhat = 0
    #Get the features for the prediction date
    for i in list_testing_days_prior:
        X_test = get_features_for_prediction_date(df, prediction_date, i)
        X_test_avg = X_test.mean().values.reshape(1, -1)  #Normalising values
        yhat = model.predict(X_test_avg)
        list_prediction.append(yhat[0])

        #mae
        mae = mean_absolute_error([df.loc[prediction_date, 'Adj Close']], [yhat[0]])
        list_error_prediction.append(mae)
        print(f"for {i} prediction days before 2020-04-01, Predicted Adj Closing value is {yhat[0]}, MAE:{mae}")
    return list_prediction

print("Actual value", df.loc[prediction_date, 'Adj Close'])
print(predictions(df, list_testing_days_prior))
print("error", list_error_prediction)

