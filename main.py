import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split

# create data ,weights and bias
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02
# set manual seed
torch.manual_seed(42)
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
# print(X), print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
plt.scatter(X, y, cmap=plt.cm.RdYlBu)
plt.show()


def plot_predictions(train_data=X_train, train_labels=y_train,
                     test_data=X_test, test_labels=y_test, Predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training Data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing Data')
    plt.legend()
    plt.show()
    if Predictions is not None:
        plt.scatter(test_data, Predictions, c='r', s=4, label='Predictions')
        plt.show()


# print(plot_predictions(X_train, y_train, X_test, y_test))
"""Create a Linear Model by Subclassing nn.module"""


class LinearRegressionV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1))

    def forward(self, x: torch.Tensor):
        return self.linear_layer(x)


model_1 = LinearRegressionV2()
# print(model_1.state_dict())
loss_fn = nn.L1Loss()
optimizers = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

""" Training Loop"""
epochs = 300
epoch_count = []
loss_values = []
test_loss_values = []
for epochs in range(epochs):
    model_1.train()
    y_pred = model_1(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizers.zero_grad()
    loss.backward()
    optimizers.step()
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)
        if epochs % 10 == 0:
            epoch_count.append(epochs)
            loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(f"Epoch:{epochs}|Loss:{loss}|Test_loss:{test_loss} ")
        # print(plot_predictions(Predictions=y_pred))
