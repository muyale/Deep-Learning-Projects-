import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchmetrics
from torchmetrics import Accuracy
from torch import nn
from torch.utils.data import DataLoader, Dataset

fifa_matches = pd.read_csv("C:\\Datasets\\FIFA WORLD CUP\\wcmatches.csv")
# print(fifa_matches.head())
# print(fifa_matches.isnull().sum())
fifa_matches['winning_team'] = fifa_matches['winning_team'].dropna()
fifa_matches['losing_team'] = fifa_matches['losing_team'].dropna()
fifa_matches.drop(['win_conditions', 'date', 'month', 'dayofweek', 'city', 'year'], axis=1, inplace=True)


# print(fifa_matches.isnull().sum())


def insight(column, df=fifa_matches):
    """This function takes in the column we are studying and returns the most prevalent items
    Args:
        column(str): This is the attributed column that we can study
        df(DataFrame) : Our  dataframe
        """
    print(df[column].value_counts())
    print(df[column].unique())
    return df[column].value_counts().head(10)


def proportion(column, df=fifa_matches):
    """This function returns the highest appearing value in each column and displays its proportion
    Args:
        column(str) ; This is the column that represents the attribute of our study
        df(DataFrame): This is our  data frame
    Returns :
        """
    my_dict = dict(df[column].value_counts().head(10))
    results = my_dict.values()
    for result in results:
        highest_proportion = [(result / sum(my_dict.values())) * 100]
        return print(f"This is the highest proportion{highest_proportion}%")


def important_statistics(column, df=fifa_matches):
    """A function to return the most vital information for every column"""
    print(insight(column=column))
    print(proportion(column=column))


print(fifa_matches.columns)
# print(fifa_matches.head(10))
# print(important_statistics('stage'))
# print(important_statistics('country'))
# print(important_statistics('winning_team'))
# print(important_statistics('losing_team'))

""" Germany has appeared the most in the world cup 102 times and Brazil seconds it,it has appeared 86 times
 Brazil has won the most times 76 to be precise which is 19% of the total matches in the tournaments
 Mexico has lost the most matches in the tournament ,29  which represents 13% of the total matches played
 Argentina has  appeared 38 times ,won 47 times and lost 24 games  which is an interesting statistics"""
print(fifa_matches['outcome'].value_counts().head(5))
fifa_matches['outcome'] = fifa_matches['outcome'].replace({'H': 1, 'A': 2, 'D': 3})
fifa_matches['outcome'] = fifa_matches['outcome'].astype('int')
# 1 :Home Team,2:Away Team.],3:Draw
print(important_statistics('outcome'))
""" Of the total games played ,46 % :429 times teams  listed as home in the fixtures won,302 lost
For example if the match is listed as Brazil vs Argentina ,Brazil is most likely to win """
columns_to_dummy = ['country', 'stage', 'winning_team', 'losing_team', 'home_team', 'away_team']
Country = pd.get_dummies(fifa_matches, 'country', drop_first=True)
stage = pd.get_dummies(fifa_matches, 'stage', drop_first=True)
winning_team = pd.get_dummies(fifa_matches, 'winning_team', drop_first=True)
losing_team = pd.get_dummies(fifa_matches, 'losing_team', drop_first=True)
home_team = pd.get_dummies(fifa_matches, 'home_team', drop_first=True)
away_team = pd.get_dummies(fifa_matches, 'away_team', drop_first=True)

fifa_matches.drop(columns_to_dummy, axis=1, inplace=True)
fifa_matches = pd.concat([fifa_matches, away_team, losing_team, winning_team, stage, Country], axis=1)
# print(fifa_matches.describe())

X = fifa_matches.drop(['outcome'], axis=1).to_numpy()
y = fifa_matches['outcome'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_test = X_test.reshape(1, X_test.shape[1])
# X_train = X.reshape(1, X_train.shape[1])
y_test = y_test
y_test = torch.from_numpy(y_test).type(torch.float32)


# USING DATA LOADER SO THAT WE CAN USE BATCHES
class Data(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


data_set = Data()
train_loader = DataLoader(dataset=data_set, batch_size=900)
print(data_set.x[1:10])


class WorldClassifier(nn.Module):
    def __init__(self, d_in, h, d_out):
        super(WorldClassifier, self).__init__()
        self.linear_1 = nn.Linear(d_in, h)
        self.linear_2 = nn.Linear(h, d_out)

    def forward(self, x: torch.Tensor):
        x = torch.from_numpy(x).type(torch.float32)

        x = torch.sigmoid(self.linear_1(x))
        x = self.linear_2(x)
        x = torch.fft.fft2(x)
        return x

    def __len__(self):
        return self.__len__()


input_dim = X.shape[1]
hidden_dim = 100
output_dim = 3

world_cup = WorldClassifier(input_dim, hidden_dim, output_dim)
torch.manual_seed(42)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=world_cup.parameters(), lr=0.1)
print(world_cup.state_dict())
epochs = 200
for epoch in range(epochs):
    for X, y in train_loader:

        world_cup.train()
        y_pred = torch.argmax(world_cup(X_train).softmax(dim=1))
        loss = loss_fn(y, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.inference_mode():
            world_cup.eval()
            test_pred = world_cup(X_test).softmax(dim=1)
            test_loss = world_cup(y, test_pred)
            if epoch % 10:
                print(f"Epoch{epoch}|loss{loss}|test loss|{test_loss}")
