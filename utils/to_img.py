import os
from torchvision import datasets, transforms
from PIL import Image

mnist = datasets.MNIST(root='./data', train=True, download=False)
save_path = './mnist_images'
os.makedirs(save_path, exist_ok=True)

for idx, (img, label) in enumerate(mnist):
    label_folder = os.path.join(save_path, str(label))
    os.makedirs(label_folder, exist_ok=True)
    img.save(os.path.join(label_folder, f'{idx}.png'))
