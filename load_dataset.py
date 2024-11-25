import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Define transformations for resizing and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Download CIFAR-10 dataset
full_trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
full_testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

# Function to select the first N samples per class
def get_first_n_samples(full_dataset, samples_per_class):
    class_counts = {i: 0 for i in range(10)}  # Initialize class counts
    selected_indices = []

    for idx, (_, label) in enumerate(full_dataset):
        if class_counts[label] < samples_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
        if all(count == samples_per_class for count in class_counts.values()):
            break

    return Subset(full_dataset, selected_indices)

# Use the first 500 training and 100 test images per class
trainset = get_first_n_samples(full_trainset, samples_per_class=500)
testset = get_first_n_samples(full_testset, samples_per_class=100)

# Create data loaders
train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
test_loader = DataLoader(testset, batch_size=32, shuffle=False)

print(f"Training Set: {len(trainset)} images")
print(f"Test Set: {len(testset)} images")
