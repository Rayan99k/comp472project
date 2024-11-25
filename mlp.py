import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import joblib  # For saving and loading models

# Define the MLP Models
class ShortenedMLP(nn.Module):
    def __init__(self):
        super(ShortenedMLP, self).__init__()
        self.layer1 = nn.Linear(50, 512)
        self.activation1 = nn.ReLU()
        self.output_layer = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.output_layer(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(50, 512)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 512)
        self.batch_norm = nn.BatchNorm1d(512)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.batch_norm(x)
        x = self.activation2(x)
        x = self.layer3(x)
        return x


class ExtendedMLP(nn.Module):
    def __init__(self, base_model, additional_layers):
        super(ExtendedMLP, self).__init__()
        self.base_model = base_model
        self.additional_layers = nn.Sequential(*additional_layers)

    def forward(self, x):
        x = self.base_model(x)
        x = self.additional_layers(x)
        return x


# Save and Load Functions for Models
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_model(model_class, filename):
    model = model_class()
    model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")
    return model


# Training Function
def train_model(model, train_features, train_labels, criterion, optimizer, num_epochs=20):
    train_features = train_features.view(-1, 50).float()
    train_labels = train_labels.long()
    model.train()
    for epoch in range(num_epochs):
        outputs = model(train_features)
        loss = criterion(outputs, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# Evaluation Function
def evaluate_model(model, test_features, test_labels):
    model.eval()
    with torch.no_grad():
        outputs = model(test_features)
        _, predicted = torch.max(outputs, 1)

    predictions = predicted.cpu().numpy()
    labels = test_labels.cpu().numpy()

    accuracy = accuracy_score(labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, predictions))
    print("\nClassification Report:")
    print(classification_report(labels, predictions))


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # Load data
    train_features, train_labels = torch.load("train_features.pt")
    test_features, test_labels = torch.load("test_features.pt")

    # Convert to NumPy arrays for PCA
    pca = PCA(n_components=50)
    train_features_pca = pca.fit_transform(train_features.numpy())
    test_features_pca = pca.transform(test_features.numpy())

    # Convert back to tensors
    train_features = torch.tensor(train_features_pca, dtype=torch.float32)
    test_features = torch.tensor(test_features_pca, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss()

    # Shortened MLP
    shortened_model_path = "models/shortened_mlp.pth"
    shortened_mlp = ShortenedMLP()
    if os.path.exists(shortened_model_path):
        load_existing = input(f"Shortened MLP '{shortened_model_path}' exists. Load it? (y/n): ").strip().lower()
        if load_existing == 'y':
            shortened_mlp = load_model(ShortenedMLP, shortened_model_path)
        else:
            optimizer = optim.SGD(shortened_mlp.parameters(), lr=0.01, momentum=0.9)
            train_model(shortened_mlp, train_features, train_labels, criterion, optimizer)
            save_model(shortened_mlp, shortened_model_path)
    else:
        optimizer = optim.SGD(shortened_mlp.parameters(), lr=0.01, momentum=0.9)
        train_model(shortened_mlp, train_features, train_labels, criterion, optimizer)
        save_model(shortened_mlp, shortened_model_path)
    evaluate_model(shortened_mlp, test_features, test_labels)

    # Base MLP
    base_model_path = "models/mlp.pth"
    mlp_model = MLP()
    if os.path.exists(base_model_path):
        load_existing = input(f"Base MLP '{base_model_path}' exists. Load it? (y/n): ").strip().lower()
        if load_existing == 'y':
            mlp_model = load_model(MLP, base_model_path)
        else:
            optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)
            train_model(mlp_model, train_features, train_labels, criterion, optimizer)
            save_model(mlp_model, base_model_path)
    else:
        optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)
        train_model(mlp_model, train_features, train_labels, criterion, optimizer)
        save_model(mlp_model, base_model_path)
    evaluate_model(mlp_model, test_features, test_labels)

    # Extended MLP
    extended_model_path = "models/extended_mlp.pth"
    additional_layers = [nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 10)]
    extended_mlp = ExtendedMLP(base_model=mlp_model, additional_layers=additional_layers)

    if os.path.exists(extended_model_path):
        load_existing = input(f"Extended MLP '{extended_model_path}' exists. Load it? (y/n): ").strip().lower()
        if load_existing == 'y':
            # Recreate the model with arguments and load the state dict
            extended_mlp = ExtendedMLP(base_model=mlp_model, additional_layers=additional_layers)
            extended_mlp.load_state_dict(torch.load(extended_model_path))
            print(f"Extended MLP loaded from {extended_model_path}")
        else:
            optimizer = optim.SGD(extended_mlp.parameters(), lr=0.01, momentum=0.9)
            train_model(extended_mlp, train_features, train_labels, criterion, optimizer)
            save_model(extended_mlp, extended_model_path)
    else:
        optimizer = optim.SGD(extended_mlp.parameters(), lr=0.01, momentum=0.9)
        train_model(extended_mlp, train_features, train_labels, criterion, optimizer)
        save_model(extended_mlp, extended_model_path)

    evaluate_model(extended_mlp, test_features, test_labels)



