import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)       # Weight

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y)**2

print(f"Loss = {loss}")

# backward pass
loss.backward()
print(f"First gradient after the forward and backward pass: {w.grad}")