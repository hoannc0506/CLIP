import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Define transformations to apply to the dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
])

# Load CIFAR-10 validation dataset
valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define classes in the dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to show an image and save it
def imshow_and_save(img, label_name, filename):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    print("image shape", npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(label_name)
    plt.savefig(filename)  # Save image
    plt.close()            # Close the figure to prevent displaying it

# Make directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Get some random validation images
images, labels = valset[2000]  # Get the first image and its label from the validation set

# Show images and save them
label_name = classes[labels]
imshow_and_save(images, label_name, f"results/{label_name}.png")

# Print label
print('Label:', label_name)
