import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

import matplotlib.pyplot as plt

# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using Device: {device}")

# =========================
# TRANSFORMS
# =========================

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================
# DATASETS
# =========================

train_dataset = datasets.ImageFolder(
    root='train',
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root='test',
    transform=transform
)

# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# =========================
# CLASSES
# =========================

classes = train_dataset.classes

print("Classes:", classes)

# =========================
# VISUALIZE TRAIN IMAGES
# =========================

images, labels = next(iter(train_loader))

fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i in range(5):

    image = images[i].squeeze()

    label = labels[i].item()

    axes[i].imshow(image, cmap='gray')

    axes[i].set_title(classes[label])

    axes[i].axis("off")

plt.show()

# =========================
# CNN MODEL
# =========================

class EmotionCNN(nn.Module):

    def __init__(self):
        super(EmotionCNN, self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        self.fc_layers = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 6 * 6, 512),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512, len(classes))
        )

    def forward(self, x):

        x = self.conv_layers(x)

        x = self.fc_layers(x)

        return x

# =========================
# MODEL
# =========================

model = EmotionCNN().to(device)

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================
# TRAINING
# =========================

epochs = 10

train_accuracies = []

for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader)

    for images, labels in loop:

        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Stats
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")

        loop.set_postfix(
            loss=loss.item(),
            accuracy=accuracy
        )

    train_accuracies.append(accuracy)

    print(f"\nEpoch Loss: {running_loss:.4f}")

# =========================
# TESTING
# =========================

model.eval()

correct = 0
total = 0

all_images = []
all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        # Save first 5 images
        for i in range(len(images)):

            if len(all_images) < 5:

                all_images.append(images[i].cpu())

                all_predictions.append(predicted[i].cpu().item())

                all_labels.append(labels[i].cpu().item())

test_accuracy = 100 * correct / total

print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# =========================
# ACCURACY GRAPH
# =========================

plt.figure(figsize=(8, 5))

plt.plot(range(1, epochs + 1), train_accuracies, marker='o')

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.grid(True)

plt.show()

# =========================
# VISUALIZE PREDICTIONS
# =========================

fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i in range(5):

    image = all_images[i].squeeze()

    predicted_label = classes[all_predictions[i]]

    actual_label = classes[all_labels[i]]

    axes[i].imshow(image, cmap='gray')

    axes[i].set_title(
        f"Pred: {predicted_label}\nActual: {actual_label}"
    )

    axes[i].axis("off")

plt.show()

# =========================
# SAVE MODEL
# =========================

torch.save(model.state_dict(), "emotion_model.pth")

print("\nModel Saved Successfully!")
