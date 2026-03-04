"""
train_eqcnn.py — Training script for Q3.c.4: Equivariant CNN experiments.

Trains and compares two models on imbalanced rotated MNIST (28x28):
  1. Standard CNN (baseline)
  2. Your rotation-equivariant CNN

Usage:
    python train_eqcnn.py                          # CPU (slow)
    python train_eqcnn.py --device cuda             # GPU
    python train_eqcnn.py --device cuda --epochs 50 # custom epochs

"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from models import GEquivariantCNN


# =============================================================================
# Label mapping for orientation prediction
# =============================================================================

LABEL_MAP = {0: 0, 90: 2, 180: 3, 270: 1}
LABEL_TO_ORIENT = {v: k for k, v in LABEL_MAP.items()}  


# =============================================================================
# Standard CNN baseline (for orientation prediction on 28x28, 4-class)
# =============================================================================

class StandardCNN(nn.Module):
    """Simple CNN baseline for 28x28 orientation prediction (4 classes)."""
    def __init__(self, num_classes=4, n_ch=32):
        super().__init__()
        # 28x28 input -> conv(5,pad=2) -> 28x28 -> pool(2) -> 14x14
        # -> conv(5,pad=2) -> 14x14 -> pool(2) -> 7x7
        self.net = nn.Sequential(
            nn.Conv2d(1, n_ch, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(n_ch, n_ch * 2, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(n_ch * 2 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Dataset — 28x28 rotated MNIST
# =============================================================================

class RotatedMNIST(Dataset):
    """
    Rotated MNIST dataset (28x28) for orientation prediction.

    Each MNIST digit is rotated by 0, 90, 180, 270 degrees.
    Labels are assigned via LABEL_MAP (regular representation: degree // 90).

    Args:
        train: if True, use training split
        imbalanced: if True, subsample 180-degree examples (train only)
        data_dir: path to MNIST data
        n_per_orient: number of examples per orientation (train only)
        n_180: number of 180-degree examples (train only, when imbalanced)
    """
    def __init__(self, train=True, imbalanced=True, data_dir='./data',
                 n_per_orient=None, n_180=None):
        mnist = datasets.MNIST(data_dir, train=train, download=True)
        images = mnist.data.float() / 255.0  # [N, 28, 28]

        all_imgs, all_labels = [], []
        for k in range(4):
            degree = k * 90
            rotated = torch.rot90(images, k=k, dims=[-2, -1])
            label_idx = LABEL_MAP[degree]
            all_imgs.append(rotated)
            all_labels.append(torch.full((len(images),), label_idx, dtype=torch.long))

        self.images = torch.cat(all_imgs).unsqueeze(1)  # [4N, 1, 28, 28]
        self.labels = torch.cat(all_labels)

        if train:
            indices = []
            for label_idx in range(4):
                oi = torch.where(self.labels == label_idx)[0]
                perm = torch.randperm(len(oi))
                orient = LABEL_TO_ORIENT[label_idx]
                if orient == 180 and imbalanced:
                    n = n_180 if n_180 else 20
                else:
                    n = n_per_orient if n_per_orient else len(oi)
                indices.append(oi[perm[:min(n, len(oi))]])
            idx = torch.cat(indices)
            self.images = self.images[idx]
            self.labels = self.labels[idx]

        for label_idx in range(4):
            n = (self.labels == label_idx).sum().item()
            orient = LABEL_TO_ORIENT[label_idx]
            tag = "Train" if train else "Test"
            print(f"  {tag} label={label_idx} ({orient:3d}deg): {n:6d}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(imgs)
    return loss_sum / total, correct / total


def evaluate(model, loader, device):
    """Evaluate model, returning overall and per-orientation accuracy."""
    model.eval()
    cp, tp = [0] * 4, [0] * 4
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            for label_idx in range(4):
                m = labels == label_idx
                tp[label_idx] += m.sum().item()
                cp[label_idx] += (preds[m] == labels[m]).sum().item()
    acc = sum(cp) / max(sum(tp), 1)
    per = {}
    for label_idx in range(4):
        orient = LABEL_TO_ORIENT[label_idx]
        per[orient] = cp[label_idx] / max(tp[label_idx], 1)
    return acc, per


def run(model, train_loader, test_loader, device, epochs, lr, name):
    """Train a model and return final test accuracy."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 68}")
    print(f"{name}  ({npar:,} params)")
    print(f"{'=' * 68}")
    for ep in range(epochs):
        loss, tacc = train_epoch(model, train_loader, opt, device)
        if (ep + 1) % 10 == 0 or ep == 0:
            acc, per = evaluate(model, test_loader, device)
            print(f"Ep {ep + 1:3d} | loss {loss:.4f} | train {tacc:.3f} | "
                  f"test {acc:.3f} | 0deg:{per[0]:.3f} 90deg:{per[90]:.3f} "
                  f"180deg:{per[180]:.3f} 270deg:{per[270]:.3f}")
    return evaluate(model, test_loader, device)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Q3.c.4: Train and compare CNN vs Equivariant CNN on rotated MNIST"
    )
    parser.add_argument('--device', default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--n-train', type=int, default=2000,
                        help='Number of training examples per orientation')
    parser.add_argument('--n-180', type=int, default=20,
                        help='Number of 180-degree training examples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Label mapping ----
    print(f"\nLabel mapping: {LABEL_MAP}")
    print(f"Reverse: {LABEL_TO_ORIENT}")

    # ---- Datasets (28x28) ----
    print(f"\n--- Training Set (imbalanced, {args.n_train}/orient, {args.n_180} for 180deg) ---")
    train_ds = RotatedMNIST(train=True, n_per_orient=args.n_train, n_180=args.n_180)
    print("\n--- Test Set (balanced) ---")
    test_ds = RotatedMNIST(train=False, imbalanced=False)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    results = {}

    # ---- 1. Standard CNN (28x28) ----
    torch.manual_seed(args.seed)
    m1 = StandardCNN()
    a1, p1 = run(m1, train_ld, test_ld, device, args.epochs, args.lr, "Standard CNN (28x28)")
    results['Standard CNN'] = (a1, p1)

    # ---- 2. G-Equivariant CNN ----
    torch.manual_seed(args.seed)
    print("\nBuilding GEquivariantCNN (computing equivariant bases -- may take a moment)...")
    m3 = GEquivariantCNN()
    a3, p3 = run(m3, train_ld, test_ld, device, args.epochs, args.lr,
                 "G-Equivariant CNN (28x28)")
    results['Equivariant CNN'] = (a3, p3)

    # ---- Summary ----
    print("\n" + "=" * 72)
    print(f"SUMMARY  (train: {args.n_train}/orient, 180deg: {args.n_180} examples)")
    print("=" * 72)
    print(f"{'Model':<30} {'All':>7} {'0deg':>7} {'90deg':>7} {'180deg':>7} {'270deg':>7}")
    print("-" * 72)
    for name, (acc, per) in results.items():
        print(f"{name:<30} {acc:>7.4f} {per[0]:>7.4f} {per[90]:>7.4f} "
              f"{per[180]:>7.4f} {per[270]:>7.4f}")
    print("=" * 72)
    print(f"\n180deg has only {args.n_180} training examples vs {args.n_train} for others.")


if __name__ == "__main__":
    main()
