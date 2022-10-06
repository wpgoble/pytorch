import torch

x_0 = torch.randn(3)  # Create a tensor with n random values
print(f"x_0 = {x_0}")

# to calculate the gradients of some function w.r.t x we need to include the 
# requires_grad arguement. When we print this, pytorch tracks that it requires 
# the gradient. Whenever we do operations with this tensor, pytorch will create 
# a computation graph.
x_1 = torch.randn(3, requires_grad=True)
print(f"x_1 = {x_1}")

y = x_1 + 2     # This creates a computational graph
print(f"x_1 + 2 = {y}")        # grad function AddBackwards

z = y**2 * 2
print(f"y**2 * 2 = {z}")        # grad function == MulBackwards

z = z.mean()
print(f"z.mean() = {z}")        # grad function == MeanBackwards

# To calculate gradients call tensor.backward()
z.backward()    # dz/dx 
print(f"x_1.grad = {x_1.grad}")

# If we didn't use requires_grad=True, then we would get an errror that the 
# tensor does not require grad and does not have a grad_fn
#y_2 = x_0 - 2
#y_2.backward()
#print(x_0.grad)

# If z is not a scalar value, then when we call backward we must provide it a 
# vector
z = y**2 * 2
v = torch.tensor([1e-1, 1.0, 1e-2], dtype=torch.float32)
z.backward(v)
print(f"z = {z}")

# How to prevent torch from tracking the history 
## Option 1: call requires_grad_(False)
## Option 2: detach()
## Option 3: with torch.no_grad():

a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
c = torch.randn(3, requires_grad=True)

print(f"a =\t\t\t\t{a}")
a.requires_grad_(False)     # modifies variable in place
print(f"a.requires_grad_(False)=\t{a}")

print(f"b = {b}")
d = b.detach()
print(f"b.detach() = {d}")

print(f"c = {c}")
with torch.no_grad():
    e = c + 2
    print(f"c + 2 = {e}")