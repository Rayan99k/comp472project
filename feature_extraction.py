import torchvision.models as models
import torch
import numpy as np
from load_dataset import test_loader, train_loader

# Load pre-trained ResNet-18
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last layer
model.eval()

# Extract features for training and testing datasets
def extract_features(loader, model):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs).squeeze(-1).squeeze(-1)  # Flatten feature vectors
            features.append(outputs)
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)

train_features, train_labels = extract_features(train_loader, model)
test_features, test_labels = extract_features(test_loader, model)

print(f"Train Features Shape: {train_features.shape}")
print(f"Test Features Shape: {test_features.shape}")

torch.save((train_features, train_labels), "train_features.pt")
torch.save((test_features, test_labels), "test_features.pt")
np.save("train_features.npy", train_features.numpy())
np.save("train_labels.npy", train_labels.numpy())
np.save("test_features.npy", test_features.numpy())
np.save("test_labels.npy", test_labels.numpy())