# src/test.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Path / import ayarı: test.py src içinde kalacak
# ------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(SRC_DIR))

# ------------------------------------------------------------
# Proje içi importlar
# ------------------------------------------------------------
# Eğer dosyanın adı dataset.py ise şu satırı:
# import ship_dataset as ds
# yerine:
# import dataset as ds
import ship_dataset as ds  # :contentReference[oaicite:1]{index=1}
from model import build_model  # src/model.py


def resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (PROJECT_DIR / pp).resolve()


def load_ckpt_any(ckpt_path: Path, device: torch.device):
    """
    Checkpoint iki tip olabilir:
      A) {'model_state_dict': ..., 'class_to_idx': ..., 'idx_to_class': ..., 'img_size': ..., 'run_name': ...}
      B) direkt state_dict
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        class_to_idx = ckpt.get("class_to_idx")
        idx_to_class = ckpt.get("idx_to_class")
        img_size = int(ckpt.get("img_size", 224))
        run_name = ckpt.get("run_name", ckpt_path.stem)
        return state, class_to_idx, idx_to_class, img_size, run_name
    return ckpt, None, None, 224, ckpt_path.stem


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path, title: str):
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def predict_loader(model, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in tqdm(loader, desc="predict", leave=False):
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.append(preds)
        y_true.append(yb.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


@torch.no_grad()
def predict_images_only(model, df_images: pd.DataFrame, tfm, device: torch.device, batch_size: int = 32):
    """
    Etiketsiz test için: df_images içinde filepath var.
    """
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, df: pd.DataFrame, transform):
            self.df = df.reset_index(drop=True)
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i: int):
            row = self.df.iloc[i]
            from PIL import Image
            img = Image.open(row["filepath"]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, row["image"]

    ds_ = DummyDataset(df_images, tfm)
    loader = DataLoader(ds_, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_images, all_preds = [], []
    for xb, names in tqdm(loader, desc="submit", leave=False):
        xb = xb.to(device)
        logits = model(xb)
        pred_idx = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(pred_idx)
        all_images.extend(list(names))

    return all_images, np.concatenate(all_preds)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True, help="Örn: outputs/best_noaug.pth")

    # mode:
    # eval   -> metrik üretir (etiketli veri gerekir)
    # submit -> etiketsiz test csv'den tahmin üretir (Kaggle submission)
    ap.add_argument("--mode", choices=["eval", "submit"], default="eval")

    # eval için iki seçenek:
    # 1) train.csv'den val split ile test (split=val)
    # 2) etiketli ayrı csv ile test (split=testcsv_labeled)
    ap.add_argument("--split", choices=["val", "testcsv_labeled", "testcsv_unlabeled"], default="val")

    ap.add_argument("--train_csv", default="data/train.csv", help="train.csv yolu (val split için)")
    ap.add_argument("--test_csv", default=None, help="test csv yolu (labeled/unlabeled için)")
    ap.add_argument("--img_dir", default="data/images", help="images klasörü")

    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # çıktılar
    ap.add_argument("--out_prefix", default=None, help="Çıktı dosyaları prefix'i (opsiyonel)")
    ap.add_argument("--submission_path", default="outputs/submission.csv",
                    help="mode=submit çıktısı (image,category)")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = resolve(args.ckpt)
    img_dir = resolve(args.img_dir)
    train_csv = resolve(args.train_csv)
    test_csv = resolve(args.test_csv) if args.test_csv else None

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {ckpt_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"img_dir bulunamadı: {img_dir}")

    # ---- checkpoint ----
    state, class_to_idx, idx_to_class, img_size, run_name = load_ckpt_any(ckpt_path, device)

    # ---- class maps (train.csv üzerinden garanti) ----
    # ship_dataset.load_dataframe train.csv'den class_name üretir (category->name map)
    # Böylece idx_to_class / class_to_idx sabitlenir.
    if not train_csv.exists():
        raise FileNotFoundError(f"train_csv bulunamadı: {train_csv}")

    cfg = ds.DataConfig(
        project_dir=PROJECT_DIR,
        data_dir=DATA_DIR,
        csv_path=train_csv,
        img_dir=img_dir,
        out_dir=OUT_DIR,
    )
    df_train_all = ds.load_dataframe(cfg)  # expects image+category :contentReference[oaicite:2]{index=2}
    classes_sorted, class_to_idx_ref, idx_to_class_ref = ds.build_class_maps(df_train_all)  # :contentReference[oaicite:3]{index=3}

    # checkpoint içindekiler varsa bile, referansı train.csv'den alıyoruz (tutarlılık)
    class_to_idx = class_to_idx_ref
    idx_to_class = idx_to_class_ref
    num_classes = len(class_to_idx)

    # ---- model ----
    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(state)
    print(f"Loaded ckpt: {ckpt_path.name} | run_name={run_name} | img_size={img_size} | num_classes={num_classes}")
    print("Device:", device)

    # ---- transforms (ship_dataset ile aynı) ----
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    _, val_tfms = ds.get_transforms(img_size=img_size, mean=mean, std=std, augment=False)  # :contentReference[oaicite:4]{index=4}

    # çıktılar prefix
    prefix = args.out_prefix or f"{run_name}_{args.split}"
    if args.mode == "submit":
        prefix = args.out_prefix or f"{run_name}_submit"

    # ------------------------------------------------------------
    # MODE: SUBMIT (etiketsiz test csv)
    # ------------------------------------------------------------
    if args.mode == "submit":
        if test_csv is None:
            raise ValueError("mode=submit için --test_csv vermelisin.")
        if not test_csv.exists():
            raise FileNotFoundError(f"test_csv bulunamadı: {test_csv}")

        df_test = pd.read_csv(test_csv)
        if "image" not in df_test.columns:
            raise ValueError(f"Etiketsiz test csv içinde 'image' kolonu olmalı. Gelen kolonlar: {list(df_test.columns)}")

        # filepath üret
        df_test["filepath"] = df_test["image"].astype(str).apply(lambda x: str((img_dir / x).as_posix()))

        # resim var mı kontrol (ilk birkaçında)
        for p in df_test["filepath"].head(5):
            if not Path(p).exists():
                raise FileNotFoundError(f"Test görseli bulunamadı: {p}")

        images, pred_idx = predict_images_only(
            model, df_test[["image", "filepath"]], val_tfms, device, batch_size=args.batch_size
        )

        # idx -> category id (train.csv'de category 1..5 idi)
        # class_name sırası: sorted(["Cargo","Carrier","Cruise","Military","Tanker"]) olabilir.
        # Kaggle submission genelde category ID ister (1..5).
        name_to_id = {"Cargo": 1, "Military": 2, "Carrier": 3, "Cruise": 4, "Tanker": 5}
        pred_names = [idx_to_class[int(i)] for i in pred_idx]
        pred_category = [name_to_id[n] for n in pred_names]

        sub = pd.DataFrame({"image": images, "category": pred_category})
        out_path = resolve(args.submission_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_path, index=False)
        print(f"✅ Submission yazıldı: {out_path}")
        return

    # ------------------------------------------------------------
    # MODE: EVAL
    # ------------------------------------------------------------
    if args.split == "val":
        # train.csv üzerinden stratify split (train.py ile uyumlu)
        from sklearn.model_selection import train_test_split

        train_df, val_df = train_test_split(
            df_train_all,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=df_train_all["class_name"],
        )

        val_ds = ds.ShipDataset(val_df, class_to_idx, transform=val_tfms)  # :contentReference[oaicite:5]{index=5}
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        y_true, y_pred = predict_loader(model, val_loader, device)

        acc = accuracy_score(y_true, y_pred)
        labels = [idx_to_class[i] for i in range(num_classes)]
        rep = classification_report(y_true, y_pred, target_names=labels, digits=4)
        cm = confusion_matrix(y_true, y_pred)

        print(f"✅ Accuracy: {acc:.4f} (N={len(y_true)})")

        (OUT_DIR / f"classification_report_{prefix}.txt").write_text(rep, encoding="utf-8")
        save_json({"accuracy": float(acc)}, OUT_DIR / f"metrics_{prefix}.json")
        plot_confusion_matrix(cm, labels, OUT_DIR / f"confusion_matrix_{prefix}.png",
                              title=f"Confusion Matrix ({prefix})")

        print(f"📁 Kaydedildi: outputs/classification_report_{prefix}.txt")
        print(f"📁 Kaydedildi: outputs/metrics_{prefix}.json")
        print(f"📁 Kaydedildi: outputs/confusion_matrix_{prefix}.png")
        return

    if args.split == "testcsv_labeled":
        if test_csv is None:
            raise ValueError("split=testcsv_labeled için --test_csv vermelisin.")
        if not test_csv.exists():
            raise FileNotFoundError(f"test_csv bulunamadı: {test_csv}")

        df_test = pd.read_csv(test_csv)

        # Labeled test csv şu kolonlardan biriyle uyumlu olmalı:
        # - image + category (train.csv gibi)
        # - veya image + class_name
        cols = {c.lower(): c for c in df_test.columns}
        if "image" not in cols:
            raise ValueError(f"Labeled test csv içinde 'image' kolonu olmalı. Gelen: {list(df_test.columns)}")

        if "category" in cols:
            # train.csv formatı gibi -> class_name üretelim
            df_test["category"] = df_test[cols["category"]]
            id_to_name = {1: "Cargo", 2: "Military", 3: "Carrier", 4: "Cruise", 5: "Tanker"}
            df_test["class_name"] = df_test["category"].map(id_to_name)
            if df_test["class_name"].isna().any():
                raise ValueError("test csv'de category map edilemeyen değer var.")
        elif "class_name" in cols:
            df_test["class_name"] = df_test[cols["class_name"]]
        else:
            raise ValueError("Labeled test csv'de 'category' veya 'class_name' olmalı.")

        df_test["filepath"] = df_test[cols["image"]].astype(str).apply(lambda x: str((img_dir / x).as_posix()))

        test_ds = ds.ShipDataset(df_test, class_to_idx, transform=val_tfms)  # :contentReference[oaicite:6]{index=6}
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        y_true, y_pred = predict_loader(model, test_loader, device)

        acc = accuracy_score(y_true, y_pred)
        labels = [idx_to_class[i] for i in range(num_classes)]
        rep = classification_report(y_true, y_pred, target_names=labels, digits=4)
        cm = confusion_matrix(y_true, y_pred)

        print(f"✅ Test Accuracy: {acc:.4f} (N={len(y_true)})")

        (OUT_DIR / f"classification_report_{prefix}.txt").write_text(rep, encoding="utf-8")
        save_json({"accuracy": float(acc)}, OUT_DIR / f"metrics_{prefix}.json")
        plot_confusion_matrix(cm, labels, OUT_DIR / f"confusion_matrix_{prefix}.png",
                              title=f"Confusion Matrix ({prefix})")

        print(f"📁 Kaydedildi: outputs/classification_report_{prefix}.txt")
        print(f"📁 Kaydedildi: outputs/metrics_{prefix}.json")
        print(f"📁 Kaydedildi: outputs/confusion_matrix_{prefix}.png")
        return

    if args.split == "testcsv_unlabeled":
        raise ValueError("Etiketsiz test için mode=submit kullan. Örn: --mode submit --split testcsv_unlabeled")


if __name__ == "__main__":
    main()
