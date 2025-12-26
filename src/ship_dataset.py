# src/dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


@dataclass
class DataConfig:
    project_dir: Path
    data_dir: Path
    csv_path: Path
    img_dir: Path
    out_dir: Path


def resolve_project_paths(project_dir: Path | str | None = None):
    if project_dir is None:
        project_dir = Path(__file__).resolve().parents[1]
    else:
        project_dir = Path(project_dir).resolve()

    data_dir = project_dir / "data"
    out_dir  = project_dir / "outputs"
    csv_path = data_dir / "train.csv"
    img_dir  = data_dir / "images"

    return project_dir, data_dir, out_dir, csv_path, img_dir
    """
    Proje kökünü bulur ve data yollarını üretir.
    Beklenen yapı:
      project/
        data/
          train.csv
          images/
    """
    if project_dir is None:
        # Bu dosya src/dataset.py olduğuna göre proje kökü = src'nin bir üstü
        project_dir = Path(__file__).resolve().parents[1]
    else:
        project_dir = Path(project_dir).resolve()

    data_dir = project_dir / "data"
    csv_path = data_dir / "train.csv"
    img_dir = data_dir / "images"
    out_dir = project_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Basit doğrulama
    missing = []
    if not data_dir.exists():
        missing.append(f"DATA_DIR yok: {data_dir}")
    if not csv_path.exists():
        missing.append(f"train.csv yok: {csv_path}")
    if not img_dir.exists():
        missing.append(f"images yok: {img_dir}")

    if missing:
        raise FileNotFoundError("\n".join(missing))

    return DataConfig(
        project_dir=project_dir,
        data_dir=data_dir,
        csv_path=csv_path,
        img_dir=img_dir,
        out_dir=out_dir,
    )


def load_dataframe(cfg: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)

    # Kaggle ship dataset genelde: image, category
    # Sende de bu şekildeydi.
    cols = {c.lower(): c for c in df.columns}
    if "image" not in cols or "category" not in cols:
        raise ValueError(f"CSV kolonları beklenmiyor. Gelen kolonlar: {list(df.columns)}")

    image_col = cols["image"]
    label_col = cols["category"]

    # filepath üret
    df["filepath"] = df[image_col].astype(str).apply(lambda x: str((cfg.img_dir / x).as_posix()))
    df["category"] = df[label_col]

    # kategori -> sınıf adı eşlemesi (senin önceki mapping’in)
    id_to_name = {1: "Cargo", 2: "Military", 3: "Carrier", 4: "Cruise", 5: "Tanker"}
    df["class_name"] = df["category"].map(id_to_name)

    # Eğer map edilemeyen varsa uyar
    if df["class_name"].isna().any():
        bad = df[df["class_name"].isna()]["category"].value_counts()
        raise ValueError(f"category map edilemeyen değerler var: {bad.to_dict()}")

    return df


def build_class_maps(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    classes_sorted = sorted(df["class_name"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return classes_sorted, class_to_idx, idx_to_class


class ShipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: Dict[str, int], transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img_path = row["filepath"]
        img = Image.open(img_path).convert("RGB")
        y = self.class_to_idx[row["class_name"]]
        if self.transform:
            img = self.transform(img)
        return img, y


def get_transforms(img_size: int, mean, std, augment: bool):
    """
    augment=False -> sadece resize+normalize
    augment=True  -> basit augmentation + normalize
    """
    if augment:
        train_tfms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_tfms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tfms, val_tfms


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    train_tfms,
    val_tfms,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    train_ds = ShipDataset(train_df, class_to_idx, transform=train_tfms)
    val_ds = ShipDataset(val_df, class_to_idx, transform=val_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader



# -------------------------------
# DataFrame'e filepath ekleyen yardımcı fonksiyon
# -------------------------------

from pathlib import Path
import pandas as pd

def resolve_paths(df: pd.DataFrame, img_dir: Path, img_col: str = "image") -> pd.DataFrame:
    """
    CSV'deki image isimlerinden tam görsel yolunu üretir.
    Örn: data/images/abc.jpg
    """
    df = df.copy()
    df["filepath"] = df[img_col].apply(lambda x: str(Path(img_dir) / x))
    return df
