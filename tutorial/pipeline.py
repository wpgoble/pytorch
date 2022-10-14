"""
Steps in the pipeline
1. Design model (input, output size, forward pass)
2. Construct the loss and optimizer
3. Training loop
    -- forward pass: compute prediction
    -- backward pass: calculate gradients
        -- update weights
"""

import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
test_tensor = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

# needs an input and output size
input_size = n_features
output_size = n_features
    
model = LinearRegression(input_size, output_size)

print(f"Prediction before training: f(5) = {model(test_tensor).item():.3f}")

lrn_rate = 1e-2
n_iters = 123
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lrn_rate)   # Stochastic Gradient Descent

for epoch in range(n_iters):
    y_pred = model(X)
    
    l = loss(Y, y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.3f}')

print(f'Prediction after training: f(5) = {model(test_tensor).item():.3f}')