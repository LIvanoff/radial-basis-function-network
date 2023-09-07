import torch
import torch.nn as nn
from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt
from predict import predict
from sklearn.model_selection import train_test_split

x = torch.rand(3000)
x = x * 25.0 - 10.0
y = torch.sin(x)


x = x.unsqueeze_(1)
y = y.unsqueeze_(1)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.10)

model = RBF(1, 100)
model.train()
optimizer = torch.optim.Adam(params=model.parameters())
criterion = nn.MSELoss()

history_train = []
history_val = []
for epoch in range(2000):
    optimizer.zero_grad()
    pred = model.forward(x_train)
    loss_t = criterion(pred, y_train)
    loss_t.backward()
    optimizer.step()
    history_train.append(loss_t.item())

    with torch.no_grad():
        model.eval()
        pred = model.forward(x_val)
        loss_v = criterion(pred, y_val)
        history_val.append(loss_v.item())

    if epoch % 100 == 0:
        print('epoch: ' + str(epoch) + ' train loss: ' + str(round(loss_t.item(), 3)) + ' val loss: ' + str(round(loss_v.item(), 3)))


fig, axe = plt.subplots(1, 2, figsize=(16, 4))
axe[0].scatter(x_val.numpy(), y_val.numpy(), marker='o')
pred = predict(model, x_val)
axe[0].scatter(x_val.numpy(), pred.detach().numpy(), marker='o', alpha=0.8)
axe[1].plot(np.arange(len(history_train)), history_train, label='train')
axe[1].plot(np.arange(len(history_val)), history_val, label='val')
plt.legend()
plt.show()
