"""
Train binary fatigue CNN on data/train/{drowsy,notdrowsy} → best_model.pth

Improvements over baseline ~76% val:
- Stratified train/val split (stable, class-balanced holdout)
- Conv + BatchNorm (faster convergence, better small-data generalization)
- AdamW + weight decay, OneCycleLR scheduler
- Label smoothing + optional mixup (reduces overfitting)
- Stronger augmentation, gradient clipping
- More epochs with early stopping on val loss

Export: run export_model_onnx.py after training (architecture must match).
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
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
    """Conv-BN-ReLU blocks + classifier (must match export_model_onnx.py)."""

    def __init__(self, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
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


def mixup_data(x, y, alpha=0.35, device=None):
    """Mixup batch; returns mixed x, y_a, y_b, lam."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    if batch_size < 2:
        return x, y, y, 1.0
    perm = torch.randperm(batch_size, device=device)
    mixed = lam * x + (1.0 - lam) * x[perm]
    return mixed, y, y[perm], lam


def evaluate(model, loader, crit, device, tta=True):
    """Validation with optional TTA (horizontal flip average) — more stable val accuracy."""
    model.eval()
    tot, cor, loss_sum = 0, 0, 0.0
    crit_hard = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if tta:
                logits = 0.5 * (model(x) + model(torch.flip(x, dims=[3])))
            else:
                logits = model(x)
            loss_sum += crit_hard(logits, y).item()
            cor += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)
    n = max(len(loader), 1)
    return loss_sum / n, cor / max(tot, 1)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((72, 72)),
            transforms.RandomResizedCrop((64, 64), scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(18),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.04),
            transforms.RandomGrayscale(p=0.08),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.12, scale=(0.02, 0.12), ratio=(0.5, 1.5)),
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
    n = len(full)
    indices = np.arange(n)
    labels = np.array([full.samples[i][1] for i in range(n)], dtype=np.int64)

    try:
        tr_i, va_i = train_test_split(
            indices,
            test_size=0.2,
            stratify=labels,
            random_state=42,
            shuffle=True,
        )
    except ValueError:
        tr_i, va_i = train_test_split(
            indices, test_size=0.2, random_state=42, shuffle=True
        )
    tr_i, va_i = tr_i.tolist(), va_i.tolist()

    tr_ds = FatigueDataset("data/train", train_tf)
    va_ds = FatigueDataset("data/train", val_tf)
    tr_sub = Subset(tr_ds, tr_i)
    va_sub = Subset(va_ds, va_i)

    sampler = build_sampler(tr_ds, tr_i)
    batch_size = 32
    if sampler is None:
        tr_ld = DataLoader(
            tr_sub, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
        )
    else:
        tr_ld = DataLoader(
            tr_sub,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
        )
    va_ld = DataLoader(va_sub, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FatigueDetectionCNN(dropout=0.25).to(device)
    prev = Path("best_model.pth")
    if prev.exists():
        try:
            model.load_state_dict(
                torch.load(prev, map_location=device, weights_only=True)
            )
            print("Warm-start from best_model.pth (only if architecture matches).")
        except RuntimeError:
            print("best_model.pth incompatible (e.g. old no-BN weights); training from scratch.")

    epochs = 40
    patience = 12
    lr_max = 2.5e-3
    wd = 5e-3
    opt = optim.AdamW(model.parameters(), lr=lr_max / 25, weight_decay=wd)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    steps_per_epoch = max(len(tr_ld), 1)
    sched = optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr_max,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    hist = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }
    best_val = 0.0
    best_val_loss = float("inf")
    stale = 0

    use_mixup = True
    mixup_alpha = 0.35

    for ep in range(epochs):
        model.train()
        loss_sum, cor, tot = 0.0, 0, 0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if use_mixup and model.training and x.size(0) > 1:
                x_m, y_a, y_b, lam = mixup_data(x, y, mixup_alpha, device)
                logits = model(x_m)
                loss = lam * crit(logits, y_a) + (1.0 - lam) * crit(logits, y_b)
                pred = logits.argmax(1)
                cor += (
                    lam * (pred == y_a).float() + (1.0 - lam) * (pred == y_b).float()
                ).sum().item()
            else:
                logits = model(x)
                loss = crit(logits, y)
                pred = logits.argmax(1)
                cor += (pred == y).sum().item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()

            loss_sum += loss.item()
            tot += y.size(0)

        tr_loss = loss_sum / max(len(tr_ld), 1)
        tr_acc = cor / max(tot, 1)
        va_loss, va_acc = evaluate(model, va_ld, crit, device)
        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)
        hist["lr"].append(float(opt.param_groups[0]["lr"]))

        print(
            f"Epoch {ep+1:02d}/{epochs}  train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}  "
            f"val_loss={va_loss:.4f}  lr={opt.param_groups[0]['lr']:.2e}"
        )

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ★ saved best_model.pth (val_acc={best_val:.4f})")

        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stop: no val_loss improvement for {patience} epochs.")
                break

    with open("training_history.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    print(f"\nDone. Best val_acc={best_val:.4f}  (stratified split, BN+AdamW+OneCycle+mixup)")
    print("Run: python export_model_onnx.py")


if __name__ == "__main__":
    main()
