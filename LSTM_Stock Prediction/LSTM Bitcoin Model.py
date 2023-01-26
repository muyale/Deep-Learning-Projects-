#!/usr/bin/env python
# coding: utf-8

# In[124]:


# importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import cycle
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score,r2_score
from sklearn.metrics import mean_poisson_deviance,mean_gamma_deviance,accuracy_score
from sklearn.preprocessing import MinMaxScaler


# In[125]:


# Read in our data
bitcoin_df = pd.read_csv("C:\\Users\EDGAR MUYALE DAVIES\\Downloads\\BTC-USD.csv")
bitcoin_df.head()
bitcoin_df = bitcoin_df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'})
bitcoin_df.head()


# In[126]:


binance_df = pd.read_csv("C:\\Users\EDGAR MUYALE DAVIES\\Downloads\\BNB-USD.csv")
binance_df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'},inplace=True)
binance_df.head()


# In[127]:


cardano_df = pd.read_csv("C:\\Users\EDGAR MUYALE DAVIES\\Downloads\\BNB-USD.csv")
cardano_df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'},inplace=True)
cardano_df


# In[128]:


dataframes = [binance_df,bitcoin_df,cardano_df]
def important_insight():
   
    dataframes = [binance_df,bitcoin_df,cardano_df]
    for dataframe in dataframes:
        print (f"Important_information for{dataframe}|Which is{dataframe.info()}")
        print(dataframe.describe())
        print(dataframe.isnull().sum())


# In[129]:


important_insight()


# In[159]:


# Preprocessing by filling NA values using forward fill
dataframes
def fill_all_null():
    for dataframe in dataframes:
        dataframe = dataframe.fillna(method='ffill')
        print(dataframe.isnull().sum())


# In[131]:


fill_all_null()


# In[132]:


binance_df['date'] = pd.to_datetime(binance_df.date)
binance_df.head().style.set_properties(subset=['date','close'], **{'background-color': 'pink'})
bitcoin_df['date'] = pd.to_datetime(bitcoin_df.date)
bitcoin_df.head().style.set_properties(subset=['date','close'], **{'background-color': 'skyblue'})
cardano_df['date']= pd.to_datetime(cardano_df.date)
cardano_df.head().style.set_properties(subset=['date','close'], **{'background-color': 'yellow'})
bitcoin_df.head()
bitcoin_df['date'] = pd.to_datetime(bitcoin_df.date)
bitcoin_df.head()


# In[133]:



bitcoin_df.head()


# In[134]:


cardano_df.head()


# In[135]:


sns.lineplot(data=bitcoin_df,x=bitcoin_df.date,y=bitcoin_df.adj_close)
plt.title('Bitcoin Changing overtime')


# In[136]:


# Creating Subplots using plt.subplots
fig = plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.plot(bitcoin_df['date'],bitcoin_df['adj_close'],color='red')
plt.title('Bitcoin close prices')
plt.subplot(2,2,2)
plt.plot(binance_df['date'],binance_df['adj_close'],color='green')
plt.title('Binance close prices')
plt.subplot(2,2,3)
plt.plot(cardano_df['date'],cardano_df['adj_close'],color='blue')
plt.title('Cardano close prices')


# In[137]:


last_year_bitcoin_df = bitcoin_df[bitcoin_df['date']>'09-2020'] 
last_year_binance_df = binance_df[cardano_df['date']>'09-2020'] 
last_year_cardano_df = cardano_df[cardano_df['date']>'09-2020'] 



# In[138]:


fig=plt.figure(figsize=(15,12))
fig.suptitle('2021 close prices of Bitcoin Cardano and Binance')
plt.subplot(4,1,1)
plt.plot(last_year_bitcoin_df['date'],last_year_bitcoin_df['adj_close'],color='black')
plt.legend('BT')
plt.subplot(4,1,2)
plt.plot(last_year_binance_df['date'],last_year_binance_df['adj_close'],color='blue')
plt.legend('N')
plt.subplot(4,1,3)
plt.plot(last_year_cardano_df['date'],last_year_cardano_df['adj_close'],color='yellow')
plt.legend('C')
# From the figure all the three financial assets seem to have increased for the one year period


# In[139]:


# Since THE STOCK PRICES ARE HIGHLY VOLATILE WE WILL COMPUTE THEIR MOVING AVERAGES
fig = plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.plot(bitcoin_df['date'],bitcoin_df['adj_close'].rolling(50).mean())
plt.plot(bitcoin_df['date'],bitcoin_df['adj_close'].rolling(200).mean())
plt.title('Bitcoin Prices moving Average')
plt.subplot(2,2,2)
plt.plot(binance_df['date'],binance_df['adj_close'].rolling(50).mean(),color='red')
plt.plot(binance_df['date'],binance_df['adj_close'].rolling(200).mean(),color='blue')
plt.title('Binance Prices Moving Averages')
plt.subplot(2,2,3)
plt.plot(cardano_df['date'],cardano_df['adj_close'].rolling(50).mean(),color='green')
plt.plot(cardano_df['date'],cardano_df['adj_close'].rolling(200).mean(),color='black')
plt.title('Cardano Prices moving averages')


# In[142]:


# Creating a dataframe for close_price and date
close_df = bitcoin_df[['date','close']]
close_df = close_df[close_df['date'] > '2020-09-13']
close_df.shape
close_stock = close_df.copy()
close_df


# In[141]:



del close_df['date']
scaler=MinMaxScaler(feature_range=(0,1))
close_df=scaler.fit_transform(np.array(close_df).reshape(-1,1))
print(close_df.shape)


# In[148]:


X=close_stock['date']
y = close_stock['close']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.7,random_state =101)


# In[151]:


# Visualizing our test and train data
fig = plt.figure(figsize=(15,10))
plt.subplot(3,1,1)
sns.lineplot(x=X_train,y=y_train,color ='black')
plt.title('Train Data')
plt.legend('TrainData')
plt.subplot(3,1,2)
sns.lineplot(x=X_test,y=y_test)
plt.title('Test Data')


# In[155]:



"""from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=1000)
my_model.fit(X_train, y_train, verbose=False)
predictions =my_model.predict(X_test)"""


import torch
from torch import nn
"""This is A stock prediction project based on the LTSM Architecture"""

"""LTSM os iseful for time series problems"""
#we will slice the dta we intend to use
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1),copy=False)
price = bitcoin_df[['Close']]
price =  scaler.fit_transform(price['Close'].values.reshape(-1,1))

"""Using the sliding window method"""


def split_data(stock,lookback):
    data_raw = stock
    data= []
    #We will create all sequences of lenght
    for index in range(len(data_raw)-lookback):
        data.append(data_raw[index:index+lookback])
        data=np.array(data);
        test_set_size = int(np.round(0.2*data.shape[0]));
        train_set_size = int(np.round(data.shape[0]-(test_set_size)));
        x_train = data[:train_set_size,:-1,:]
        y_train = data[:train_set_size,:-1,:]
        x_test = data[train_set_size:,:-1]
        y_test = data[train_set_size:,:-1]
        return [x_train,y_train,x_test,y_test]


lookback = 20
x_train, y_train, x_test, y_test = split_data(price, lookback)

# We convert everything into tensors
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_ltsm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_ltsm = torch.from_numpy(y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

#some common hyper parameters for LTSM models
input_dim =1
hidden_dim = 32
num_layers = 2
output_dim = 1
epochs = 40

"""Now we create an LTSM model using nn.Module"""

class LTSM(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim,hidden_dim,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out,(hn,cn)=self.lstm(x,(h0.detach(),c0.detach()))
        out = self.fc(out[:,-1,:])
        return  out
model = LTSM(input_dim=input_dim,hidden_dim=hidden_dim,output_dim=output_dim,num_layers=num_layers)
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.01)

for epoch in range(epochs):
    y_pred_train = model(x_train)
    loss = criterion(y_pred_train,y_train_ltsm)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch{epoch}|loss{loss}")

print(y_pred_train)

Epoch1|loss0.9292709827423096
Epoch2|loss0.5563279986381531
Epoch3|loss0.12340367585420609
Epoch4|loss0.1146305501461029
Epoch5|loss0.15774765610694885
Epoch6|loss0.05938250198960304
Epoch7|loss0.0036829274613410234
Epoch8|loss0.0068739126436412334
Epoch9|loss0.029254185035824776
Epoch10|loss0.044182006269693375
Epoch11|loss0.04515630751848221
Epoch12|loss0.03564384579658508
Epoch13|loss0.021798303350806236
Epoch14|loss0.009230555966496468
Epoch15|loss0.0016302377916872501
Epoch16|loss0.00018168942187912762
Epoch17|loss0.0035774202551692724
Epoch18|loss0.008797431364655495
Epoch19|loss0.012627349235117435
Epoch20|loss0.013215030543506145
Epoch21|loss0.010691466741263866
Epoch22|loss0.006601659115403891
Epoch23|loss0.002786633325740695
Epoch24|loss0.0004924993263557553
Epoch25|loss4.89144804305397e-05
Epoch26|loss0.001030556159093976
Epoch27|loss0.0026399074122309685
Epoch28|loss0.004082124214619398
Epoch29|loss0.004813792649656534
Epoch30|loss0.004642962012439966
Epoch31|loss0.003707845462486148
Epoch32|loss0.0023746893275529146
Epoch33|loss0.0010923842201009393
Epoch34|loss0.00023994283401407301
Epoch35|loss4.722076482721604e-06
Epoch36|loss0.00032936144270934165
Epoch37|loss0.0009519036975689232
Epoch38|loss0.0015288225840777159
Epoch39|loss0.0017876336351037025
tensor([[-1.0421]], grad_fn=<AddmmBackward0>)

Process finished with exit code 0



# In[ ]:




