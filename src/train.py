# src/train.py
from __future__ import annotations

import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# -------------------------------------------------
# Proje yolları
# -------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "outputs"

sys.path.append(str(SRC_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Local imports
# -------------------------------------------------
from ship_dataset import (
    DataConfig,
    load_dataframe,
    build_class_maps,
    get_transforms,
    make_loaders,
)
from model import build_model

# -------------------------------------------------
# Ayarlar
# -------------------------------------------------
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
VAL_SIZE = 0.2
LR = 3e-4

NUM_WORKERS = 0
PIN_MEMORY = False

CSV_PATH = DATA_DIR / "train.csv"
IMG_DIR = DATA_DIR / "images"

# -------------------------------------------------
# Seed
# -------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------------------------------
# DataFrame yükleme
# -------------------------------------------------
cfg = DataConfig(
    project_dir=PROJECT_DIR,
    data_dir=DATA_DIR,
    out_dir=OUT_DIR,
    csv_path=CSV_PATH,
    img_dir=IMG_DIR,
)

df = load_dataframe(cfg)

# filepath üret
df["filepath"] = df["image"].apply(lambda x: str(IMG_DIR / x))

missing = df[~df["filepath"].apply(lambda p: Path(p).exists())]
print("Toplam kayıt:", len(df))
print("Eksik görsel sayısı:", len(missing))

# -------------------------------------------------
# Train / Val split (stratify)
# -------------------------------------------------
train_df, val_df = train_test_split(
    df,
    test_size=VAL_SIZE,
    random_state=SEED,
    stratify=df["class_name"],
)

print("Train size:", len(train_df), "Val size:", len(val_df))
print("\nTrain dağılımı:\n", train_df["class_name"].value_counts())
print("\nVal dağılımı:\n", val_df["class_name"].value_counts())

# -------------------------------------------------
# Sınıf haritaları
# -------------------------------------------------
classes, class_to_idx, idx_to_class = build_class_maps(df)
num_classes = len(classes)

print("Classes:", classes)
print("Num classes:", num_classes)

# -------------------------------------------------
# Transforms (NO-AUG)
# -------------------------------------------------
train_tfms, val_tfms = get_transforms(
    img_size=IMG_SIZE,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    augment=False,
)

train_loader, val_loader = make_loaders(
    train_df=train_df,
    val_df=val_df,
    class_to_idx=class_to_idx,
    train_tfms=train_tfms,
    val_tfms=val_tfms,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

# -------------------------------------------------
# Model / Optimizer
# -------------------------------------------------
model = build_model(num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

print("Model hazır")

# -------------------------------------------------
# Train & Eval fonksiyonları
# -------------------------------------------------
def accuracy_from_logits(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, yb)

    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def eval_one_epoch(model, loader):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    for xb, yb in tqdm(loader, desc="val", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, yb)

    return total_loss / len(loader), total_acc / len(loader)

# -------------------------------------------------
# Predict + rapor
# -------------------------------------------------
@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    y_true, y_pred = [], []

    for xb, yb in tqdm(loader, desc="predict", leave=False):
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()

        y_true.append(yb.numpy())
        y_pred.append(preds)

    return np.concatenate(y_true), np.concatenate(y_pred)


def save_report_and_cm(y_true, y_pred, run_name: str):
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # classification report
    rep_txt = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    (OUT_DIR / f"classification_report_{run_name}.txt").write_text(rep_txt, encoding="utf-8")

    rep_dict = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, output_dict=True
    )
    with open(OUT_DIR / f"metrics_{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(rep_dict, f, ensure_ascii=False, indent=2)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix ({run_name})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"confusion_matrix_{run_name}.png", dpi=200)
    plt.close()

# -------------------------------------------------
# Training loop
# -------------------------------------------------
def run_training(run_name: str, train_loader, val_loader):
    best_val_acc = -1.0
    best_path = OUT_DIR / f"best_{run_name}.pth"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader)
        va_loss, va_acc = eval_one_epoch(model, val_loader)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        scheduler.step(va_acc)

        print(
            f"[{run_name}] Epoch {epoch}/{EPOCHS} | "
            f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f} "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "img_size": IMG_SIZE,
                    "run_name": run_name,
                },
                best_path,
            )
            print(f"✅ Best saved: {best_path} (val_acc={best_val_acc:.4f})")

    # en iyi modeli yükle ve rapor üret
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_pred = predict_all(model, val_loader)
    save_report_and_cm(y_true, y_pred, run_name)

    return history, best_val_acc, best_path

# -------------------------------------------------
# 1) NO-AUG
# -------------------------------------------------
history_noaug, best_noaug, path_noaug = run_training(
    run_name="noaug",
    train_loader=train_loader,
    val_loader=val_loader,
)

print("NO-AUG best val acc:", best_noaug)

# -------------------------------------------------
# 2) AUG
# -------------------------------------------------
train_tfms_aug, val_tfms_aug = get_transforms(
    img_size=IMG_SIZE,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    augment=True,
)

train_loader_aug, val_loader_aug = make_loaders(
    train_df=train_df,
    val_df=val_df,
    class_to_idx=class_to_idx,
    train_tfms=train_tfms_aug,
    val_tfms=val_tfms_aug,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

# modeli sıfırdan kur
model = build_model(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

history_aug, best_aug, path_aug = run_training(
    run_name="aug",
    train_loader=train_loader_aug,
    val_loader=val_loader_aug,
)

print("AUG best val acc:", best_aug)

# -------------------------------------------------
# Karşılaştırma
# -------------------------------------------------
compare = {
    "noaug": {"best_val_acc": float(best_noaug), "best_path": str(path_noaug)},
    "aug": {"best_val_acc": float(best_aug), "best_path": str(path_aug)},
    "settings": {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "img_size": IMG_SIZE,
        "lr": LR,
        "val_size": VAL_SIZE,
    },
}

with open(OUT_DIR / "compare_runs.json", "w", encoding="utf-8") as f:
    json.dump(compare, f, ensure_ascii=False, indent=2)

print("Karşılaştırma kaydedildi:", OUT_DIR / "compare_runs.json")
