import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)

# --- EXPERIMENT 2: NOISY GRADIENTS WITH BATCH SIZE = 1 ---
BATCH_SIZE = 1  # 1 image per batch (Maximum Gradient Variance!)
LEARNING_RATE = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Use subset of 2,000 images so batch_size=1 runs quickly on CPU
train_dataset = datasets.MNIST(root='./data/raw', train=True, download=True, transform=transform)
subset_indices = range(2000)
train_subset = torch.utils.data.Subset(train_dataset, subset_indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

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

step_losses = []

print(f"--- Task 6.4: Experiment #2 - Noisy Gradients (batch_size={BATCH_SIZE}) ---\n")

model.train()
for step, (image, label) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(image)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    step_losses.append(loss.item())

    # Log every 200 steps
    if (step + 1) % 400 == 0 or step == 0:
        print(f"Step [{step + 1:4d}/2000] | Step Loss: {loss.item():.4f}")

# Plotting step-by-step noisy loss trajectory
os.makedirs('experiments/run_logs', exist_ok=True)
plot_path = 'experiments/run_logs/noisy_batch1_curve.png'

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(step_losses) + 1), step_losses, label='Step Loss (batch_size=1)', color='purple', alpha=0.6, linewidth=0.8)
plt.title('EXPERIMENT: Noisy Loss Curve with Batch Size = 1')
plt.xlabel('Step Index (Individual Samples)')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(plot_path)
print(f"\nNoisy loss plot saved to '{plot_path}'!")
