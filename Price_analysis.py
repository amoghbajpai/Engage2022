
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

prices= pd.read_csv("C:/Users/amogh/Desktop/Engage_2022/tables_V2.0/Price_table.csv")
df=pd.DataFrame(prices)
df.head()

gk = df.groupby('Maker')
maker= input("Enter Maker name from the dataset: ")
a= gk.get_group(maker)
a


gk2= a.groupby('Genmodel')
model= input("Enter Model name belonging to the chosen maker from the dataset: ")
b= gk2.get_group(model)
b.index = b['Year']
b

b = b[['Year', 'Entry_price']]
b.drop('Year',axis=1,inplace=True)
b.head()

l= len(b) 
l

reshaped_b= b.values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(reshaped_b)
reshaped_b = scaler.transform(reshaped_b)


n_prev = 3  
n_next = 2

X = []
Y = []

for i in range(n_prev, len(reshaped_b) - n_next + 1):
    X.append(reshaped_b[i - n_prev: i])
    Y.append(reshaped_b[i: i + n_next])

X = np.array(X)
Y = np.array(Y)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_prev, 1)))
model.add(LSTM(units=50))
model.add(Dense(n_next))
model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=100, batch_size=n_prev, verbose=0)

last= reshaped_b[- n_prev:]  
last= last.reshape(1, n_prev, 1)

Next= model.predict(last).reshape(-1, 1)
Next= scaler.inverse_transform(Next)

b_past= b.reset_index()
b_past.index= b_past['Year']
b_past['Forecast'] = np.nan
b_past['Forecast'].iloc[-1] = b_past['Entry_price'].iloc[-1]
b_past


b_future= pd.DataFrame(index= np.arange(n_next + 1))
b_future = pd.DataFrame(columns=['Year', 'Entry_price', 'Forecast'])

arr= []
for i in range(n_next+1):
    arr.append(b_past['Year'].iloc[-1] + i)
    
b_future['Year']= arr[1: n_next + 1]   

b_future.index= b_future['Year']
b_future['Forecast'] = Next.flatten()
b_future.head()


forecasting= b_past.append(b_future).set_index('Year')
forecasting

forecasting.plot(title= 'Prices of required model and maker')



