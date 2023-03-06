import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("C:\\Datasets\\cocacola\\Coca-Cola_stock_history.csv")
print(df.head())

price = df[['Close']]
scaler = MinMaxScaler(feature_range=(-1,1),copy=False)
price = scaler.fit_transform(price.values().reshape(-1,1))
