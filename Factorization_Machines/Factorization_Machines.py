import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torch import no_grad,trunc_,normal,pow,sum
from sklearn.decomposition import TruncatedSVD
import time

df = pd.read_csv("C:\\Datasets\\imdb_top_1000.csv")
print(df.head())
print(df.isnull().sum())
df.dropna()
print("\n")
print(df.shape)
df.drop(['Poster_Link','Certificate','Gross'],axis=1,inplace=True)
#I will create a movie recommender system using Factorization machines
"""Factorization machines have an advantage in that the users can get recommendations from a cold start"""



class FactorizationMachine(nn.Module):
    def __init__(self,n,k):
        super().__init__()
        self.w0  = nn.Parameter(torch.zeros(1))# This is the initial weight
        self.bias = nn.Embedding(n,1)
        self.embeddings = nn.Embedding(n,k)
        with torch.no_grad():trunc_(normal(self.embeddings.weight,std = 0.01))# This is to speed up learning of our embeddings
        with torch.no_grad():trunc_(normal(self.bias.weight,std = 0.01))

    def forward(self,X):
        emb = self.embeddings(X)
        #Interactions in complexity of 0nk
        pow_of_sum = emb.sum(dim=1).pow(2)
        sum_of_pow = emb.pow(2).sum(dim=1)
        pairwise = (pow_of_sum-sum_of_pow).sum(1)*0.5
        bias = self.bias(X).squeeze().sum(1)
        return torch.sigmoid(self.w0 + bias + pairwise) * 5.5


features_columns = ['Series_Title','Genre','Director','Star1','Star2','Star3','Star4','Overview','Released_Year']


def get_important_features(df):
    important_features = []
    for i in range(0,df.shape[0]):
        important_features.append(df['Series_Title'][i]+df['Genre'][i]+df['Director'][i]+df['Star1'][i]+df['Star2'][i]+df['Star3'][i]+df['Star4'][i]+df['Overview'][i]+
                                  df['Released_Year'][i])
    return important_features

df['important_features'] = get_important_features(df)
print(df.head())
my_important_features = df['important_features']
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(my_important_features)
X = cosine_similarity(X)

print(X.shape) # These are the embeddings that our model nneds to learn,after which we will use Dimensionality reduction and KNN ,to try and place the learned emdeddings
truncated = TruncatedSVD(n_components=3)
components = truncated.fit_transform(X)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=3)
tnse_comp = tsne.fit_transform(X)
model = FactorizationMachine(1000, 12428)
tsne.fit_transform(model(X))

# Prepare the data for Pytorch DataLoader
class Data(Dataset):
    def __init__(self):
        self.X= torch.from_numpy(X)
        self.len = X.shape[0]
    def __getitem__(self, item):
        return self.X[index]
    def __len__(self):
        return self.len

data = Data()

data_x = torch.tensor(X)
data_y = torch.tensor(df['IMDB_Rating'])
data_set = data.TensorDataset(data_x,data_y)
bs = 1024
train_n = int(len(data_set)*0.8)
valid_n = len(data_set) - train_n
splits = [train_n,valid_n]
#assert sum(splits) == len(data_set)
train_set,dev_set = torch.utils.data.random_split(data_set,splits)
train_dataloader = data.DataLoader(train_set,batch_size=bs)
dev_dataloader = data.DataLoader(dev_set,batch_size=bs)

# Training and testing loop
def fit(iterator,model,criterion,optimizer):
    train_loss = 0
    model.train()
    for x,y in iterator:
        optimizer.zerograd()
        y_hat = model(X)
        loss = criterion(y_hat,y)
        train_loss = loss.item()*X.shape[0]
        loss.backward()
        optimizer.step()
    return train_loss/len(iterator.data_set)

def test(iterator,model,criterion,optimizer):
    train_loss = 0
    model.train()
    for x,y in iterator:
        optimizer.zerograd()
        y_hat = model(X)
        loss = criterion(y_hat,y)
        train_loss = loss.item()*X.shape[0]
        loss.backward()
        optimizer.step()
    return train_loss/len(iterator.data_set)

model = FactorizationMachine(data_x.max()+1,X.shape[0])
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
criterion = nn.MSELoss(reduction='mean')
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=7,gamma=0.03)

for epoch in range(epochs):
    start_time = time.time()
    train_loss = fit(train_dataloader,model,criterion,optimizer)
    valid_loss = test(dev_dataloader,model,criterion,optimizer)
    scheduler.step()
    secs = int(time.time()-start_time)
    print(f"Epoch{epoch}|time{secs}")
    print(f"trainloss{train_loss}|test_loss{valid_loss}")


"""This is a partially completed project,the Idea of Learning Embeddings to recommend based on Cosine Similarity is to be developed on,The main challenge I have was to convert the tensors
into Numpy Arrays before they were embedded,embedding loses data and In my project I used TSNE to try and figure out which embeddings were close to each other,
I welcome any correction and modifications for  this project,included is the IMDB Dataset"""