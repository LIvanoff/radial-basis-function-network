import torch
import torch.nn as nn
from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt
from utils.predict import predict
from sklearn.model_selection import train_test_split

x_train = torch.rand(1000)
x_train, indices1 = torch.sort(x_train)
x_train = x_train * 25.0 - 10.0
y_train = torch.sin(x_train)

x_train = x_train.unsqueeze_(1)
y_train = y_train.unsqueeze_(1)

x_val = torch.rand(50)
x_val, indices2 = torch.sort(x_val)
x_val = x_val * 25.0 - 10.0
y_val = torch.sin(x_val)

x_val = x_val.unsqueeze_(1)
y_val = y_val.unsqueeze_(1)

# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, shuffle=False)

model = RBF(1, 100)
model.train()
optimizer = torch.optim.Adam(params=model.parameters())
criterion = nn.MSELoss()

history_train = []
history_val = []

overfiting_epoch = None
ofe_num = 0
flag = True
epochs = 1000
best_acc = 10e+6
for epoch in range(epochs):
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
        if loss_v.item() < best_acc:
            best_acc = loss_v.item()
        if loss_v.item() > best_acc:
            ofe_num += 1
        else:
            ofe_num = 0
    if ofe_num == epochs / 10 and flag == True:
        overfiting_epoch = epoch - (epochs / 10 - 1)
        flag = False

    if epoch % 100 == 0:
        print('epoch: ' + str(epoch) + ' train loss: ' + str(round(loss_t.item(), 3)) + ' val loss: ' + str(round(loss_v.item(), 3)))


fig, axe = plt.subplots(1, 2, figsize=(16, 4))
axe[0].scatter(x_val.numpy(), y_val.numpy(), marker='o', c='#4169E1', alpha=0.8, label='Groud truth')  #  c='#7B68EE'
pred = predict(model, x_val)
axe[0].plot(x_val.numpy(), pred.detach().numpy(), 'r',  label='Prediction')
if overfiting_epoch is not None:
    min_lim = min(history_val) if min(history_val) < min(history_train) else min(history_train)
    max_lim = max(history_val) if max(history_val) > max(history_train) else max(history_train)
    axe[1].vlines(overfiting_epoch, min_lim, max_lim, linestyles='dashed', color='red', label='overfit')

axe[1].plot(np.arange(len(history_train)), history_train, label='train')
axe[1].plot(np.arange(len(history_val)), history_val, label='val')
axe[1].grid(0.5)
axe[0].legend()
axe[1].legend()
plt.show()
