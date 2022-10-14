import torch
import torch.nn as nn

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

# Prepare data
## generate a regression dataset
X_np, Y_np = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_np.astype(np.float32))
y = torch.from_numpy(Y_np.astype(np.float32))

# reshape the y so we have a column vector
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# Define the model
input_sz, output_sz = n_features, 1
#model = nn.Linear(input_sz, output_sz)

model = LinearRegression(input_sz, output_sz)

# loss and optimizer
lrn_rate = 1e-2
criterion = nn.MSELoss()        # returns a callable function
optimizer = torch.optim.SGD(model.parameters(), lr = lrn_rate)

# training loop
n_iters = 125
for epoch in range(n_iters):
    # forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()       # empty gradients

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

# plot
predicted = model(X).detach().numpy()

plt.plot(X_np, Y_np, 'ro', label='generated data')
plt.plot(X_np, predicted, 'b-', label='predicted')
plt.legend(loc='upper left')
plt.show()