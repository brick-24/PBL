from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision.models import resnet18
from torchvision import datasets


resnet = torch.load("trained/resnet_mnist_full (1).pth", map_location="cpu", weights_only=False)
resnet.eval()

img = Image.open('sample.png').convert('L')  # 'L' = grayscale

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.Grayscale(num_output_channels=3),  # convert 1 channel -> 3
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

for images, labels in test_loader:
    output = resnet(images)
    pred = output.argmax(dim=1).item()
    print("Pred:", pred, "Actual:", labels.item())
    if pred != labels.item():
        break