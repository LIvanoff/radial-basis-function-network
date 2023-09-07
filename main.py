import torch
import torch.nn as nn
from rbf import RBF

x_train = torch.Tensor(
    [[4.5, 5.], [4.3, 5.1], [4.5, 4.4], [5., 5.], [4.9, 4.8], [5.1, 5.], [5.3, 4.1], [5.2, 4.8], [5.1, 5.2], [4.9, 4.6],
     [4.3, 5.1], [4.2, 5.], [4.4, 4.2], [5.1, 5.3], [4.7, 4.8], [5., 5.], [5.2, 4.5], [5.3, 4.6], [5., 5.3], [4.7, 4.3],
     [6., 2.], [3., 4.5], [4., 3.], [5., 2.8], [5.5, 2.9], [6.8, 3.5], [7.1, 4.5], [6.5, 5.9], [6.3, 6.3], [4.5, 7.], [2.5, 6.],
     [6.3, 2.8], [6.1, 4.4], [3.5, 4.], [2.8, 3.], [2.9, 5.9], [3.5, 6.8], [4.5, 7.1], [5.9, 6.5], [6.2, 6.3], [4.6, 7.1], [2.3, 6.1],
     [6., 2.3], [3., 5.2], [4., 3.], [5.1, 2.4], [5.2, 2.6], [3.9, 3.5], [4.1, 2.2], [3.5, 5.9], [4.3, 6.3], [3.5, 3.], [2.5, 4.6],
     [4.1, 2.2], [3.3, 4.3], [3.5, 3.3], [2.5, 3.2], [2.9, 3.9], [3.2, 6.4], [4.3, 7.1], [5.4, 6.1], [6.1, 6.1], [4.3, 7.], [2.3, 4.1]])


y_train = torch.Tensor([[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
                        [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
                        [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                        [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                        [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                        [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]])

model = RBF(2, 5)
model.train()
optimizer = torch.optim.Adam(params=model.parameters())
criterion = nn.MSELoss()

for epoch in range(10000):
    optimizer.zero_grad()
    pred = model.forward(x_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('epoch: ' + str(epoch) + ' loss: ' + str(loss))
