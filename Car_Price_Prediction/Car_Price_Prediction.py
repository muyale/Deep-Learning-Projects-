import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

train_data = pd.read_csv("C:\\Datasets\\Car_Prediction\\train-data.csv")
test_data =  pd.read_csv("C:\\Datasets\\Car_Prediction\\test-data.csv")
train_data.drop(['Unnamed: 0','New_Price'],axis=1,inplace=True)
train_data.dropna()



def convert_mileage_data(val):
    if not pd.isnull(val):
        return val.split(' ')[0]
    return float(val)

def convert_engine_data(val):
    if not pd.isnull(val):
        return val.split(' ')[0]
    return float(val)

train_data["Engine"] = train_data["Engine"].apply(lambda val: convert_engine_data(val))

train_data["Mileage"] = train_data["Mileage"].apply(lambda val: convert_mileage_data(val))

import datetime
train_data['Total Years'] = datetime.datetime.now().year - train_data["Year"]
train_data['Engine'] = train_data['Engine'].apply(lambda val:convert_engine_data(val))
train_data.drop(['Year'],axis=1,inplace=True)


for column in train_data.columns:
    print(f"The total number of elements in {column} are :{len(train_data[column].unique())}")


cat_features = ['Location','Fuel_Type','Transmission','Owner_Type','Name','Power','Mileage','Engine']
out_features = ['Price']
from sklearn.preprocessing import LabelEncoder

lbl_encoders = {}
lbl_encoders["Fuel_Type"] = LabelEncoder()
lbl_encoders["Fuel_Type"].fit_transform(train_data["Fuel_Type"])
from sklearn.preprocessing import LabelEncoder

lbl_encoders = {}
for features in cat_features:
    lbl_encoders[features] = LabelEncoder()
    train_data[features] = lbl_encoders[features].fit_transform(train_data[features])
print(train_data.head(10))
# Converting our categorical features into Numpy array then into tensors
cat_features = np.stack([train_data['Location'],train_data['Fuel_Type'],train_data['Transmission'],train_data['Owner_Type'],train_data['Name'],train_data['Power'],
                         train_data['Mileage'],train_data['Engine']],1)
print(cat_features)
cat_features = torch.tensor(cat_features,dtype=torch.int64)
print(cat_features)

# Choosing our continous features
cont_features = []
for i in train_data.columns:
    if i in ['Location','Fuel_Type','Transmission','Owner_Type','Name','Price','Power','Mileage','Engine']:
        pass
    else :
        cont_features.append(i)
print(cont_features)
cont_features = np.stack([train_data[i].values for i in cont_features], axis=1)
cont_features = torch.tensor(cont_features,dtype=torch.float)
print(cont_features)
train_data['Price'] = train_data['Price']*1000

# Creating embedding Dimensions
y = torch.tensor(train_data['Price'].values,dtype=torch.float).reshape(-1,1)
# print(y)
print(cont_features.shape,cat_features.shape,y.shape)

cat_dims = [train_data[i].nunique() for i in ['Kilometers_Driven', 'Seats', 'Total Years']]
print(cat_dims)
embed_dims = [(i,min(50,i+1//2)) for i in cat_dims]
print(embed_dims)
import torch.nn as nn
embedding_rep = nn.ModuleList([nn.Embedding(i,e) for i,e in embed_dims])
print(embedding_rep)
embed_val = []

for i,e in enumerate(embedding_rep):
    embed_val.append(e(cat_features[:,i]))
    print(embed_val)

z = torch.cat(embed_val,dim=1)
drop_out = nn.Dropout(0.4)
z = drop_out(z)
print(z)


# Creating a Neural Network that predicts future Prices

class CarPredictor(nn.Module):
    def __init__(self,cat_dim,n_cont,layers,out_sz,p=0.04):
        super().__init__()
        embedding_dims = [(i, min(50, i + 1 // 2)) for i in cat_dims]
        self.embed_list = nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dims])
        self.drop_out = nn.Dropout(p)
        self.batch_norm = nn.BatchNorm1d(n_cont)
        layer_list = []
        for i in layer_list :
            layer_list.append(nn.Linear(n_in,i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(i))
            layer_list.append(nn.Dropout(p))
            n_in = i
        layer_list.append(nn.Linear(layers[-1],out_sz))
        self.layers = nn.Sequential( * layer_list)

    def forward(self,x_cat,x_cont):
        embeddings = []
        for i,e in enumerate(self.embed_list):
            embeddings.append(e(x_cat[:,i]))
            x = torch.cat(embeddings,1)
            x = self.drop_out(x)
            x_cont = self.batch_norm(x_cont)
            x = torch.cat([x,x_cont],axis=1)
            x = self.layers(x)
            return x
"""Before we Apply our model to any case we have to  train our model"""
model = CarPredictor(cat_dims,3,[100,50],1)
print(model.parameters())
print(model)
loss_function = nn.MSELoss()
optimizer = torch.optim.ASGD(model.parameters(),lr=0.01)

# Performing a train_test_split
batch_size = 6000
test_size = int(batch_size*0.15)
train_categorical = cat_features[:batch_size-test_size]
test_categorical = cat_features[batch_size-test_size:batch_size]
train_cont = cont_features[:batch_size-test_size]
test_cont = cont_features[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size - test_size:batch_size]

# Creating a Training and Test Loop
epochs = 1000
for i in range(epochs):
    i = i+1
    y_pred = model(train_categorical,train_cont)
    loss = loss_function(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i% 10 == 0:
         print(f"Epoch{epoch} test loss{loss}")