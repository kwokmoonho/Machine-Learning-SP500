#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:21:12 2020

@author: kwokmoonho

Stock prediction by using LSTM, KNN, and linear regression

- Using python and different machine learning algorithms to conduct predictions on the S & P 500 index. (LSTM, KNN, Linear Regression)
- Implementing the library stocker from python and compare the result.

Hypothesis:
    My hypothesis is that the first and last days of the week could potentially affect the closing price of the stock more than the other days.
"""

#import library
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from fastai.tabular import add_datepart
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
scaler = MinMaxScaler(feature_range=(0, 1))
pd.options.mode.chained_assignment = None  # default='warn'

#reading data
df = pd.read_csv('sp500.csv')
#overlook the data
df.head()

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')
plt.xlabel("Years")
plt.ylabel("S&P500 index")
plt.show()


"""
Linear Regression
"""
#sorting
data = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#create features
add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp


#split into train and validation

train = new_data[:7080]
valid = new_data[7080:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']


#implement linear regression
model = LinearRegression()
model.fit(x_train,y_train)

#make predictions and find the rmse
preds = model.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print(rms)
print("It is not a good fit by using linear regression")

#plot
#add a predictions column
valid['Predictions'] = 0
valid['Predictions'] = preds
valid.index = df[7080:].index
train.index = df[:7080].index
plt.figure(figsize=(16,8))
plt.plot(train['Close'], label='Train Data')
plt.plot(valid['Close'], label='Test Data')
plt.plot(valid['Predictions'], label='Prediction')
plt.ylabel("S&P500 index")
plt.xlabel("Years")
plt.title("S&P500 Linear Regression")
plt.legend(title='Parameter where:')
plt.show()


"""
KNN
"""
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(x_train,y_train)
preds = model.predict(x_valid)

rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print(rms)

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.figure(figsize=(16,8))
plt.plot(train['Close'], label='Train Data')
plt.plot(valid['Close'], label='Test Data')
plt.plot(valid['Predictions'], label='Prediction')
plt.ylabel("S&P500 index")
plt.xlabel("Years")
plt.title("S&P500 Linear Regression")
plt.legend(title='Parameter where:')
plt.show()


"""
LSTM
"""
#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:7080,:]
valid = dataset[7080:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#check for best units
myRMS = []
for p in range (40,60):
    model = Sequential()
    model.add(LSTM(units=p, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=p))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)
    
    #predicting values, using past 60 from the train data
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)
    
    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)
    
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    
    rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
    myRMS.append(rms)
    print(rms)
    
print("Dimensionality of the output space for different units values:")
for i in range (len(myRMS)):
    print("units = {} , rms = {}".format(40+i,myRMS[i]))
    
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=57, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=57))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)

#predicting values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
myRMS.append(rms)
print(rms)

#plotting
train = new_data[:7080]
valid = new_data[7080:]
valid['Predictions'] = closing_price
plt.figure(figsize=(16,8))
plt.plot(train['Close'], label='Train Data')
plt.plot(valid['Close'], label='Test Data')
plt.plot(valid['Predictions'], label='Prediction')
plt.xlabel("Years")
plt.ylabel("S&P500 index")
plt.title("S&P500 LSTM")
plt.legend(title='Parameter where:')
plt.show()

#zoom in
plt.figure(figsize=(16,8))
plt.plot(valid['Close'], label='Test Data')
plt.plot(valid['Predictions'], label='Prediction')
plt.xlabel("Years")
plt.ylabel("S&P500 index")
plt.title("Zoom in the test result")
plt.legend(title='Parameter where:')
plt.show()
