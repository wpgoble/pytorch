import numpy as np

# Implementing linear regression from scratch
# f(x) = w * x

# f(x) = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0             # Initial weight

# model prediction
def forward(x):
    """ From the simple linear regression formula """
    return w * x

# loss
def loss(y, y_predicted):
    """
    Calculates the MSE for our target output and the model output
    """
    loss = (y - y_predicted)**2
    return loss.mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# d/dw = 1/N 2x (w*x - y)
def gradient(x, y, y_predicted):
    """ Calculates the gradient """
    grad = np.dot(2*x, y_predicted - y)
    return grad.mean()

print(f'Prediction before training f(5) = {forward(5):.3f}')

# Training
learning_rate = 1e-2
n_iters = 15

for epoch in range(n_iters):
    # prediction (forawrd pass)
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients w.r.t. w
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 3 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f} loss = {l:.8f}')

print(f'Prediction after training f(5) = {forward(5):.3f}')