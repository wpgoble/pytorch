import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyper-parameters
input_size = 28 * 28
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 1e-3

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root="../data",
                                        train=True, 
                                        transform=transforms.ToTensor(),
                                        download=True)

test_dataset = torchvision.datasets.MNIST(root="../data",
                                        train=False,
                                        transform=transforms.ToTensor())

# data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                        batch_size = batch_size,
                                        shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                        batch_size = batch_size,
                                        shuffle = False)

# Logistic regression model
model = nn.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax interrnally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=learning_rate)

# set lists to plot data
train_accu = list()
train_losses = list()


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0
    train_correct = 0
    train_total = 0

    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        running_loss += loss.item()
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}]\tStep[{}/{}]\tLoss[{:.4f}]'
            .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    train_loss = running_loss / len(train_loader)
    acc = 100.*train_correct/train_total

    train_accu.append(acc)
    train_losses.append(train_loss)

# Test the model
# In test phase we don't need to compute gradients for memory efficiency

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the model on the 10000 test images: {}'
    .format(100 * correct / total))


plt.plot(train_accu, '-o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train'])
plt.title('Train Accuracy')
plt.show()