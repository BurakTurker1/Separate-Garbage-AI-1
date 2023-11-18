import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.models import resnet50
import torch.nn as nn
from torch.optim import Adam

def main():
    # Path to the dataset
    data_path = r'D:Separate-Garbage-AI\Data\organized'

    # Load the dataset and apply transformations
    transfm = Compose([Resize((256, 256)), ToTensor()])  # Same dimensions as VGG16
    dataset = ImageFolder(root=data_path, transform=transfm)

    # Split the dataset into training and validation subsets
    train_length = 2000
    valid_length = len(dataset) - train_length
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_length, valid_length])

    # Training data loader
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    # Validation data loader
    test_loader = DataLoader(valid_set, batch_size=64, num_workers=4)

    # Load the ResNet-50 model (use pretrained weights with pretrained=True)
    model = resnet50(pretrained=True)

    # Modify the output layer according to the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)  # Modify for your 6 classes

    # If you don't want to use CUDA, set the device to CPU
    device = 'cpu'

    # Move the model to the device
    clf = model.to(device)

    # Define optimization and loss functions
    opt = Adam(clf.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            yhat = clf(X)
            loss = loss_fn(yhat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * X.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")

    # Calculate accuracy for validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = clf(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on validation set: {accuracy:.2f}%')

    # Save model weights
    torch.save(clf.state_dict(), 'resnet50_model.pth')

if __name__ == '__main__':
    main()
