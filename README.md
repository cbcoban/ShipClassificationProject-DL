# Ship Image Classification using EfficientNetB0

## Overview
This project classifies ship images into five categories (Cargo, Military, Carrier, Cruise, Tanker) using deep learning and transfer learning with EfficientNetB0.  
Developed by **Celil Berk Çoban** as part of the graduate course *Deep Learning*.

## Dataset
Kaggle: Game of Deep Learning – Ship Dataset  
Download dataset locally and place images under `data/images/`, and `train.csv` under `data/` (not included in the repo).

## Model
- EfficientNetB0 (ImageNet weights, include_top=False)
- Input: 224×224
- Optimizer: Adam (lr=1e-3), Loss: Categorical Crossentropy
- Epochs: 25
- Callbacks: EarlyStopping, ModelCheckpoint (.keras)

## Results
- Validation Accuracy ≈ **0.884**
- Macro ROC-AUC ≈ **0.984**

### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)


## How to Run
pip install -r requirements.txt

## Türkçe Açıklama
## EfficientNetB0 ile Gemi Görüntülerinin Sınıflandırılması
## Genel Bakış

Bu proje, derin öğrenme ve transfer öğrenme yaklaşımlarını kullanarak gemi görüntülerini beş kategoriye ayırmaktadır: Cargo (Yük Gemisi), Military (Askerî), Carrier (Uçak Gemisi), Cruise (Yolcu Gemisi) ve Tanker.
Proje, Celil Berk Çoban tarafından Ankara Bilim Üniversitesi Yüksek Lisans Programı kapsamında Derin Öğrenme dersi için geliştirilmiştir.

## Veri Seti

Veri kaynağı: Kaggle – Game of Deep Learning: Ship Dataset
Bu veri seti, farklı açılardan çekilmiş 2B gemi görüntülerini içermektedir.
Veri seti bu depoda paylaşılmamıştır; indirilen görüntüler data/images/ klasörüne, train.csv dosyası ise data/ klasörüne yerleştirilmelidir.

## Model Mimarisi

EfficientNetB0 (ImageNet önceden eğitilmiş ağırlıklarıyla)
Girdi boyutu: 224×224 piksel
Optimizasyon: Adam (lr = 1e-3)
Kayıp fonksiyonu: Categorical Crossentropy
Eğitim: 25 epoch
Kullanılan callback’ler: EarlyStopping ve ModelCheckpoint (.keras)

## Sonuçlar
Doğrulama doğruluğu (Validation Accuracy): 0.884
Makro ROC-AUC skoru: 0.984
Karışıklık Matrisi

## Nasıl Çalıştırılır
pip install -r requirements.txt
jupyter notebook notebooks/GemiTürüSiniflandirmaProjesi_Odev.ipynb

## Gereksinimler

requirements.txt dosyasına bakınız.

## Lisans

MIT Lisansı © 2025 Celil Berk Çoban
