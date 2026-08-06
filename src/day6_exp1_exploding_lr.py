import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)

# --- EXPERIMENT 1: EXPLODING LEARNING RATE ---
BATCH_SIZE = 64
LEARNING_RATE = 5.0  # 5,000x larger than baseline (0.001)!
EPOCHS = 5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data/raw', train=True, download=True, transform=transform)
val_dataset   = datasets.MNIST(root='./data/raw', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1000, shuffle=False)

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
# Using SGD with huge learning rate to trigger gradient explosion
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses = []

print(f"--- Task 6.3: Experiment #1 - Exploding Learning Rate (lr={LEARNING_RATE}) ---\n")

for epoch in range(1, EPOCHS + 1):
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

# Plotting the broken experiment curve
os.makedirs('experiments/run_logs', exist_ok=True)
plot_path = 'experiments/run_logs/exploding_lr_curve.png'

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss', color='red', marker='o')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Val Loss', color='darkred', marker='s')
plt.title(f'BROKEN Experiment: Exploding LR (lr={LEARNING_RATE})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(plot_path)
print(f"\nPlot saved to '{plot_path}'!")
