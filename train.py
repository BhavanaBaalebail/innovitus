"""
Train binary fatigue CNN on data/train/{drowsy,notdrowsy} → best_model.pth
Matches export_model_onnx.py / app.py preprocessing (64×64, normalize 0.5).
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms


class FatigueDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        for img_file in sorted((self.root_dir / "drowsy").glob("*.*")):
            self.samples.append((str(img_file), 1))
        for img_file in sorted((self.root_dir / "notdrowsy").glob("*.*")):
            self.samples.append((str(img_file), 0))
        if not self.samples:
            raise ValueError(f"No images in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, torch.tensor(label, dtype=torch.long)


class FatigueDetectionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_sampler(dataset: FatigueDataset, indices):
    labels = [dataset.samples[i][1] for i in indices]
    counts = np.bincount(labels, minlength=2)
    if counts[0] == counts[1]:
        return None
    w = 1.0 / np.clip(counts, 1, None)
    sw = [w[l] for l in labels]
    return WeightedRandomSampler(
        torch.DoubleTensor(sw), num_samples=len(sw), replacement=True
    )


def evaluate(model, loader, crit, device):
    model.eval()
    tot, cor, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += crit(logits, y).item()
            cor += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)
    n = max(len(loader), 1)
    return loss_sum / n, cor / max(tot, 1)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.RandomResizedCrop((64, 64), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    full = FatigueDataset("data/train", transform=None)
    idx = np.arange(len(full))
    np.random.shuffle(idx)
    n_tr = int(0.8 * len(idx))
    tr_i, va_i = idx[:n_tr].tolist(), idx[n_tr:].tolist()

    tr_ds = FatigueDataset("data/train", train_tf)
    va_ds = FatigueDataset("data/train", val_tf)
    tr_sub = Subset(tr_ds, tr_i)
    va_sub = Subset(va_ds, va_i)

    sampler = build_sampler(tr_ds, tr_i)
    if sampler is None:
        tr_ld = DataLoader(tr_sub, batch_size=32, shuffle=True, num_workers=0)
    else:
        tr_ld = DataLoader(tr_sub, batch_size=32, sampler=sampler, num_workers=0)
    va_ld = DataLoader(va_sub, batch_size=32, shuffle=False, num_workers=0)

    model = FatigueDetectionCNN().to(device)
    prev = Path("best_model.pth")
    if prev.exists():
        try:
            model.load_state_dict(torch.load(prev, map_location=device, weights_only=True))
            print("Warm-start from best_model.pth")
        except RuntimeError:
            print("best_model.pth incompatible; training from scratch.")

    opt = optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss()
    hist = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best = 0.0
    epochs = 8

    for ep in range(epochs):
        model.train()
        loss_sum, cor, tot = 0.0, 0, 0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            cor += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)
        tr_loss = loss_sum / max(len(tr_ld), 1)
        tr_acc = cor / max(tot, 1)
        va_loss, va_acc = evaluate(model, va_ld, crit, device)
        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)
        print(f"Epoch {ep+1}/{epochs} train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  saved best_model.pth (val_acc={best:.4f})")

    with open("training_history.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    print(f"Done. Best val_acc={best:.4f}")


if __name__ == "__main__":
    main()
