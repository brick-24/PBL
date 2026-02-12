from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision.models import resnet18

resnet = torch.load("resnet_mnist_full.pth", map_location="cpu", weights_only=False)
resnet.eval()

img = Image.open('image.png').convert('L')  # 'L' = grayscale
img.resize((28,28))
img.show()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.Grayscale(num_output_channels=3),  # convert 1 channel -> 3
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

img_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 224, 224]

with torch.no_grad():
    output = resnet(img_tensor)
    probs = torch.softmax(output, dim=1)
    predicted_class = probs.argmax(dim=1).item()

print("Predicted class:", predicted_class)
for i, p in enumerate(probs[0]):
    print(f"Class {i}: {p.item()*100:.2f}%")
