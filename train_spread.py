import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data_loader import SoliDataset
from model_Spread import SoliModel

def compute_spread(X):
    # X: (B, T, 1, 32, 32)
    X = X.squeeze(2)              # -> (B, T, 32, 32)
    doppler = X.sum(dim=2)        # sum over range -> (B, T, 32)

    # normalize per (B,T) to avoid scale issues
    max_val = doppler.max(dim=2, keepdim=True).values + 1e-6
    doppler_norm = doppler / max_val

    # threshold-based width (simple, robust)
    mask = (doppler_norm > 0.3).float()  # tune 0.2–0.4 if needed
    spread = mask.mean(dim=2, keepdim=True)  # -> (B, T, 1)

    return spread

# -------- Config --------
DATA_PATH = "Your_dataset_path"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 11
DEVICE = torch.device("cpu")  # keep CPU for now

# -------- Dataset & Split --------
train_dataset = SoliDataset(DATA_PATH, allowed_sessions=[2,3,5,6,8,9])
test_dataset  = SoliDataset(DATA_PATH, allowed_sessions=[10,11,12,13])

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

# -------- Model --------
model = SoliModel(num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------- Train --------
def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Enable gradient on input
        X.requires_grad = True

        # ---- Clean forward ----
        spread = compute_spread(X).to(DEVICE)
        out = model(X, spread)
        loss = criterion(out, y)

        # ---- Get gradient wrt input ----
        loss.backward(retain_graph=True)
        grad = X.grad

        # ---- Create adversarial sample ----
        epsilon = 0.01
        X_adv = X + epsilon * grad.sign()
        X_adv = torch.clamp(X_adv, 0, 1)
        X_adv = X_adv.detach()

        # ---- Adversarial forward ----
        spread_adv = compute_spread(X_adv).to(DEVICE)
        out_adv = model(X_adv, spread_adv)
        loss_adv = criterion(out_adv, y)

        # ---- Combine losses ----
        final_loss = 0.5 * loss + 0.5 * loss_adv

        # ---- Backprop ----
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # Clear gradient on input
        X.grad = None

        total_loss += loss.item()

        preds = torch.argmax(out, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / len(train_loader), correct / total


# -------- Evaluate --------
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            spread = compute_spread(X).to(DEVICE)
            out = model(X, spread)
            preds = torch.argmax(out, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


# -------- Run --------
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    test_acc = evaluate()

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print("-" * 40)
