import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),  # convert 1 channel -> 3
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

vgg = models.vgg16(pretrained=True)  # ImageNet weights
vgg.classifier[6] = nn.Linear(4096, 10)  # change output layer for 10 MNIST classes
vgg = vgg.to(device)

resnet = models.resnet18(pretrained=True)  # ImageNet weights
resnet.fc = nn.Linear(resnet.fc.in_features, 10)  # change output layer
resnet = resnet.to(device)


def train_model(model, train_loader, test_loader, epochs=2, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100*correct/total:.2f}%")

print("Training VGG16...")
train_model(vgg, train_loader, test_loader, epochs=1)

print("Training ResNet18...")
train_model(resnet, train_loader, test_loader, epochs=1)

torch.save(vgg.state_dict(), 'vgg_mnist.pth')
torch.save(resnet.state_dict(), 'resnet_mnist.pth')
