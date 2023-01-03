import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from pathlib import Path


torch.manual_seed(42)
weights = 0.7
bias = 0.3
X = torch.arange(0, 1, 0.02)
y = torch.arange(0, 1, 0.02)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

    def forward(self, x: torch.tensor):
        return self.weight * x + self.bias


model_0 = LinearRegressionModel()
# print(model_0.state_dict())

with torch.inference_mode():
    y_pred = model_0(X_test)

print(y_pred)

loss_fn = nn.L1Loss()
# optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
# Training Loop


epochs = 200
epoch_count = []
loss_values = []
test_loss_values = []

for epochs in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    # print('Loss: ', loss)
    # print(model_0.state_dict())
    with torch.inference_mode():
        test_prediction = model_0(X_test)
        test_loss = loss_fn(test_prediction, y_test)

    if epochs % 10 == 0:
        epoch_count.append(epochs)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch:{epochs}|Loss:{loss}|Test_loss:{test_loss}",model_0.state_dict())

    plt.plot(epoch_count,  np.array(torch.tensor(loss_values).numpy()), label='Train Loss')
    plt.plot(epoch_count,  test_loss_values, label='Test Loss')
    plt.title('Training and Loss Curve')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    model_path = Path('Models')
    model_path.mkdir(parents=True,exist_ok=True)
    model_name = 'Regression Model.pth'
    model_save_path = model_path/model_name

    print(model_save_path)
    torch.save(obj=model_0.state_dict(),f=model_save_path)

