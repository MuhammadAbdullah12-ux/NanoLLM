import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set seed for reproducibility
torch.manual_seed(42)

# Configuration Parameters (Baseline)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

# 1. Preprocessing & DataLoaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data/raw', train=True, download=True, transform=transform)
val_dataset   = datasets.MNIST(root='./data/raw', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1000, shuffle=False)

# 2. Model Architecture
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

model = MNISTClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Lists to store loss values per epoch for plotting
train_losses = []
val_losses = []

print(f"--- Task 6.2: Running Baseline Experiment (lr={LEARNING_RATE}, batch_size={BATCH_SIZE}) ---\n")

for epoch in range(1, EPOCHS + 1):
    # Training Phase
    model.train()
    total_train_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation Phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# 3. Plotting the Baseline Loss Curve using Matplotlib
os.makedirs('experiments/run_logs', exist_ok=True)
plot_path = 'experiments/run_logs/baseline_loss_curve.png'

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss (Homework)', marker='o', color='blue')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Val Loss (Pop Quiz)', marker='s', color='orange')
plt.title(f'Baseline Loss Curve (lr={LEARNING_RATE}, batch_size={BATCH_SIZE})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(plot_path)
print(f"\nSUCCESS: Baseline loss curve plot saved to '{plot_path}'!")
