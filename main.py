import torch
import torch.nn as nn
from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt
from predict import predict

x_train = torch.rand(100)
x_train = x_train * 20.0 - 10.0
y_train = torch.sin(x_train)
x_train = x_train.unsqueeze_(1)
y_train = y_train.unsqueeze_(1)
model = RBF(1, 10)
model.train()
optimizer = torch.optim.Adam(params=model.parameters())
criterion = nn.MSELoss()

history = []
for epoch in range(2000):
    optimizer.zero_grad()
    pred = model.forward(x_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('epoch: ' + str(epoch) + ' loss: ' + str(loss))
    history.append(loss.item())


fig, axe = plt.subplots(1, 2, figsize=(16, 4))
axe[0].scatter(x_train.numpy(), y_train.numpy(), marker='o')
pred = predict(model, x_train)
axe[0].scatter(x_train.numpy(), pred.detach().numpy(), marker='o', alpha=0.8)
axe[1].plot(np.arange(len(history)), history, 'r')
plt.show()
