import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy

"""DEEP LEARNING CLASSIFICATION USING PYTORCH """
from sklearn.datasets import make_circles

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)
# print(f"The First five samples of X :{X[:5]}")

# print(f"The First five samples of y :{y[:5]}")
# circles = pd.DataFrame({"xi": X[:, 0], "x2": X[:, 1], "label": y})
# print(circles.head(10))

# plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu);
# plt.show()

# convert data into tensors
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)
X_Train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
torch.manual_seed(42)


# build a neural network using pytorch
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.two_linear_layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x: torch.Tensor):
        return self.two_linear_layers(x)


# instantiate our Circle model
circle_model = CircleModelV0()
# print(next(circle_model.parameters()))
print(circle_model.state_dict())

# Trying to predict based of our  x test
with torch.inference_mode():
    untrained_preds = circle_model(X_Test)
print(f"The shape of our untrained predictions is:{untrained_preds.shape}")
print(f"The first 5 untrained predictions are{untrained_preds[:5]}")
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=circle_model.parameters(), lr=0.1)


# Accuracy calculation
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = correct / len(y_pred) * 100
    return accuracy


# Training Loop
epochs = 200
for epochs in range(epochs):
    circle_model.train()
    y_logits = circle_model(X_Train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    circle_model.eval()
    with torch.inference_mode():
        test_logits = circle_model(X_Test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_accuracy = accuracy_fn(y_true=y_test, y_pred=test_pred)
        if epochs % 10 == 0:
            print(f"Epochs:{epochs}|Loss:{loss}|test_accuracy:{test_accuracy}|test loss:{test_loss}|accuracy{acc}")

