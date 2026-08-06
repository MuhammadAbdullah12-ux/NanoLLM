import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Set seed for reproducibility
torch.manual_seed(42)

# 2. Preprocessing & DataLoaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data/raw', train=True, download=True, transform=transform)
val_dataset   = datasets.MNIST(root='./data/raw', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1000, shuffle=False)

# 3. Model Architecture
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
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Task 5.1: Training & Validation
epochs = 3
print("Starting Training with Validation Loss Tracking...\n")

for epoch in range(1, epochs + 1):
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

    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()
            
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = (correct / total) * 100

    print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

# Task 5.2: Model Checkpointing
os.makedirs('checkpoints', exist_ok=True)
checkpoint_path = 'checkpoints/mnist_model.pt'

print(f"\n[Task 5.2] Saving student's brain state to '{checkpoint_path}'...")
torch.save(model.state_dict(), checkpoint_path)
print("SUCCESS: Model checkpoint saved to disk!")

# --- TASK 5.3 ADDITION: RELOADING & SINGLE-IMAGE INFERENCE ---
print("\n[Task 5.3] Testing Checkpoint Reloading & Single-Image Inference...")

# 1. Instantiate a brand-new student with an empty brain
new_model = MNISTClassifier()

# 2. Download/Load the saved brain state from USB drive into new student
new_model.load_state_dict(torch.load(checkpoint_path))
new_model.eval() # Set mode to testing (no gradient updates)

# 3. Pick 1 single sample image from test dataset
sample_image, sample_label = val_dataset[0] # First test sample

# 4. Add batch dimension (1, 1, 28, 28) using unsqueeze(0)
sample_batch = sample_image.unsqueeze(0)

# 5. Run prediction
with torch.no_grad():
    prediction_logits = new_model(sample_batch)
    predicted_digit = prediction_logits.argmax(dim=1).item()

print(f"True Label: {sample_label} | Model Predicted Digit: {predicted_digit}")
if sample_label == predicted_digit:
    print("SUCCESS: Reloaded model predicted correctly!")
