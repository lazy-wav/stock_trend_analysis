import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import yfinance as yf #alternate
import streamlit as st
# import h5py
from keras.initializers import Orthogonal

start = '2011-01-01'
end = '2022-12-31'

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker','TSLA')
df = yf.download(user_input, start, end)


#Describing Data
st.subheader('Data from 2011 - 2022')
st.write(df.describe())

#Plotting
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100ma')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100ma & 200ma')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig1 = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig1)

# Splitting data for training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


from  sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)

###
x_train =[]
y_train = []

for i in range(100, data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:i])
  y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))


model.add(LSTM(units = 60, activation = 'relu', return_sequences=True))
model.add(Dropout(0.3))


model.add(LSTM(units = 80, activation = 'relu', return_sequences=True))
model.add(Dropout(0.4))


model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))
# model.build()
model.add(Dense(units = 1))

model.summary()

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)

model.save('keras_model.keras')

#Model Loading 
model = load_model('keras_model.keras')


#Predictions
past_100_days = data_training.tail(100)

# Use the concat function to combine DataFrames
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make Predictions

y_predicted = model.predict(x_test) 

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Visualisation

st.subheader('Predicted vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


#streamlit web application of stock trend analysis
