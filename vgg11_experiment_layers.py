import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report


class VGG11(nn.Module):
    def __init__(self, num_classes=10, mode='original'):
        super(VGG11, self).__init__()
        self.features = self._make_features(mode)
        self.feature_map_size = None
        self.classifier = None
        self._initialize_weights()

        # Classifier for 224x224 images
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def _make_features(self, mode):
        layers = []
        in_channels = 3

        cfg_original = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        if mode == 'shortened':
            cfg = cfg_original[:-4]
        elif mode == 'extended':
            cfg = cfg_original + [512, 512]
        else:
            cfg = cfg_original

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
                in_channels = v

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features
        x = self.features(x)

        # Dynamically calculate the feature map size
        if self.feature_map_size is None:
            self.feature_map_size = x.size(1) * x.size(2) * x.size(3)
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_map_size, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 10),
            ).to(x.device)

        # Flatten and pass through classifier
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=20):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {running_loss / 100:.4f}, Acc: {100. * correct / total:.2f}%')
                running_loss = 0.0

        # Calculate epoch accuracy
        epoch_acc = 100. * correct / total
        scheduler.step(epoch_acc)

        if epoch_acc > best_acc:
            best_acc = epoch_acc

        print(f'Epoch {epoch + 1} complete. Accuracy: {epoch_acc:.2f}%')


def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, zero_division=0))

    return accuracy


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path, model):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    model_path = "models/vgg11_experiment.pth"

    # Data preparation
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_test, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model experiments
    for mode in ['shortened', 'original', 'extended']:
        print(f"\nTesting {mode} model...")
        model = VGG11(num_classes=10, mode=mode).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

        # Check for existing model
        mode_model_path = f"models/vgg11_{mode}.pth"
        if os.path.exists(mode_model_path):
            load_existing = input(f"Model '{mode_model_path}' exists. Load it? (y/n): ").strip().lower()
            if load_existing == 'y':
                model = load_model(mode_model_path, model)
            else:
                print("Training...")
                start_time = time.time()
                train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs)
                print(f"Training time: {time.time() - start_time:.2f} seconds")
                save_model(model, mode_model_path)
        else:
            print("Training...")
            start_time = time.time()
            train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs)
            print(f"Training time: {time.time() - start_time:.2f} seconds")
            save_model(model, mode_model_path)

        print("Evaluating...")
        evaluate_model(model, test_loader, device)