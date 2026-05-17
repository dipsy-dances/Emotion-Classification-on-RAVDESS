"""
Speech Emotion Recognition — RAVDESS Dataset
CNN on Mel Spectrograms | 8 Emotions
Run: python train.py --data_dir ./audio_speech_actors_01-24
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import json

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

EMOTION_MAP = {
    '01': 'neutral',  '02': 'calm',    '03': 'happy',  '04': 'sad',
    '05': 'angry',    '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
EMOTIONS = list(EMOTION_MAP.values())
LABEL2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2LABEL = {i: e for e, i in LABEL2IDX.items()}

SR          = 22050
DURATION    = 3.0          # seconds to crop/pad
N_MELS      = 128
N_FFT       = 2048
HOP_LENGTH  = 512
MAX_LEN     = int(SR * DURATION / HOP_LENGTH) + 1  # ~130 frames

BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────

def extract_melspec(path: str) -> np.ndarray:
    """Load wav, pad/crop to DURATION, compute log-mel spectrogram."""
    y, _ = librosa.load(path, sr=SR, duration=DURATION)
    # Pad if shorter than DURATION
    target = int(SR * DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # (128, T)

    # Normalise to [-1, 1]
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

    # Pad/crop time axis to MAX_LEN
    if log_mel.shape[1] < MAX_LEN:
        log_mel = np.pad(log_mel, ((0, 0), (0, MAX_LEN - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :MAX_LEN]

    return log_mel.astype(np.float32)   # (128, MAX_LEN)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class RAVDESSDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths   = paths
        self.labels  = labels
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def _time_shift(self, spec, max_shift=20):
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(spec, shift, axis=1)

    def _freq_mask(self, spec, max_mask=15):
        f0 = np.random.randint(0, N_MELS - max_mask)
        spec[f0:f0 + np.random.randint(1, max_mask)] = 0
        return spec

    def _time_mask(self, spec, max_mask=20):
        t0 = np.random.randint(0, MAX_LEN - max_mask)
        spec[:, t0:t0 + np.random.randint(1, max_mask)] = 0
        return spec

    def __getitem__(self, idx):
        spec = extract_melspec(self.paths[idx])     # (128, T)

        if self.augment:
            if np.random.rand() > 0.5:
                spec = self._time_shift(spec)
            if np.random.rand() > 0.5:
                spec = self._freq_mask(spec.copy())
            if np.random.rand() > 0.5:
                spec = self._time_mask(spec.copy())

        x = torch.tensor(spec).unsqueeze(0)         # (1, 128, T) — 1 channel
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ─────────────────────────────────────────────
# Model: CNN + Attention Pooling
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class EmotionCNN(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1 — 1 → 32
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResBlock(32),
            nn.MaxPool2d(2, 2),          # (32, 64, T/2)
            nn.Dropout2d(0.2),

            # Block 2 — 32 → 64
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64),
            nn.MaxPool2d(2, 2),          # (64, 32, T/4)
            nn.Dropout2d(0.2),

            # Block 3 — 64 → 128
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128),
            nn.MaxPool2d(2, 2),          # (128, 16, T/8)
            nn.Dropout2d(0.3),

            # Block 4 — 128 → 256
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # (256, 4, 4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct     += (logits.argmax(1) == y).sum().item()
        total       += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in tqdm(loader, desc="eval ", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total   += len(y)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def load_dataset(data_dir):
    paths, labels = [], []
    for actor in sorted(os.listdir(data_dir)):
        actor_path = os.path.join(data_dir, actor)
        if not os.path.isdir(actor_path) or not actor.startswith("Actor_"):
            continue
        for f in sorted(os.listdir(actor_path)):
            if not f.endswith(".wav"):
                continue
            parts = f.split("-")
            emo_code = parts[2]
            if emo_code in EMOTION_MAP:
                paths.append(os.path.join(actor_path, f))
                labels.append(LABEL2IDX[EMOTION_MAP[emo_code]])
    return paths, labels


def main(args):
    print(f"Device: {DEVICE}")
    print(f"Loading data from: {args.data_dir}")

    paths, labels = load_dataset(args.data_dir)
    print(f"Total samples: {len(paths)}")

    X_train, X_val, y_train, y_val = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    train_ds = RAVDESSDataset(X_train, y_train, augment=True)
    val_ds   = RAVDESSDataset(X_val,   y_val,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = EmotionCNN(n_classes=len(EMOTIONS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_acc, preds, gt = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"[{epoch:02d}/{EPOCHS}]  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.3f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     vl_acc,
                "emotions":    EMOTIONS,
                "label2idx":   LABEL2IDX,
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  ✓ Saved best model (val_acc={vl_acc:.3f})")

    # ── Final report ─────────────────────────────
    print("\n=== Classification Report ===")
    print(classification_report(gt, preds, target_names=EMOTIONS))

    # Confusion matrix
    cm = confusion_matrix(gt, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=EMOTIONS,
                yticklabels=EMOTIONS, cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix (best val_acc={best_val_acc:.3f})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    print(f"Confusion matrix saved.")

    # Learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="train"); ax1.plot(history["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(history["train_acc"], label="train"); ax2.plot(history["val_acc"], label="val")
    ax2.set_title("Accuracy"); ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "learning_curves.png"))

    # Save config for inference
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"sr": SR, "duration": DURATION, "n_mels": N_MELS,
                   "n_fft": N_FFT, "hop_length": HOP_LENGTH,
                   "max_len": MAX_LEN, "emotions": EMOTIONS}, f, indent=2)
    print(f"\nAll outputs saved to {args.output_dir}/")
    print(f"Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./audio_speech_actors_01-24")
    parser.add_argument("--output_dir", default="./checkpoints")
    args = parser.parse_args()
    main(args)
