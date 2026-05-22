import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Training settings
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 21 landmarks (x, y, z) = 63 inputs
INPUT_SIZE = 63
CLASSES = ["A", "B", "C", "D", "L", "O", "U", "V", "W", "Y"]
NUM_CLASSES = len(CLASSES)

# Feedforward MLP for classification
class HandGestureMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HandGestureMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Custom dataset loader for landmark files
class GestureDataset(Dataset):
    def __init__(self, data_path):
        self.features = np.load(os.path.join(data_path, "features.npy")).astype(np.float32)
        self.labels = np.load(os.path.join(data_path, "labels.npy")).astype(np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def main():
    data_dir = "./recorded_data"
    
    if not os.path.exists(data_dir) or not os.path.exists(os.path.join(data_dir, "features.npy")):
        print(f"Data directory not found at {data_dir}.")
        print("To train custom gestures:")
        print("  1. Capture joint coordinates via MediaPipe.")
        print("  2. Save features.npy and labels.npy inside recorded_data/.")
        print("  3. Run this script to export gesture_model.pth.")
        return

    print("Loading datasets...")
    dataset = GestureDataset(data_dir)
    
    # 80-20 train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = HandGestureMLP(INPUT_SIZE, NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    print("Training model...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, preds = out.max(1)
            total += y.size(0)
            correct += preds.eq(y).sum().item()

        t_loss = train_loss / len(train_set)
        t_acc = correct / total

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)

                val_loss += loss.item() * x.size(0)
                _, preds = out.max(1)
                val_total += y.size(0)
                val_correct += preds.eq(y).sum().item()

        v_loss = val_loss / len(val_set)
        v_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} Acc: {t_acc*100:.2f}% | Val Loss: {v_loss:.4f} Acc: {v_acc*100:.2f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), "gesture_model.pth")
            print("=> Saved new best model weights!")

    print(f"Finished. Best Val Accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    main()
