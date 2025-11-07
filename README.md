# Derin Öğrenme Projesi: Gemi Türü Sınıflandırması

Bu proje, Derin Öğrenme final ödevi kapsamında geliştirilmiştir. Projenin amacı, 2D gemi görsellerini 5 farklı sınıfa (Cargo, Military, Carrier, Cruise, Tanker) ayırmaktır.

## Veri Seti
Kullanılan veri seti Kaggle'da bulunan "Game of Deep Learning: Ship Datasets" adlı veri setidir.
* [Veri Seti Linki](https://www.kaggle.com/datasets/arpitjain007/game-of-deep-learning-ship-datasets)

## Yöntem
Model, TensorFlow ve Keras kütüphaneleri kullanılarak geliştirilmiştir. ImageNet üzerinde ön-eğitimli **EfficientNetB0** modeli ile Transfer Öğrenme yaklaşımı kullanılmıştır.

## Sonuçlar
[cite_start]Model, doğrulama (validation) seti üzerinde **%88.41** genel doğruluk [cite: 570-576] [cite_start]ve **0.9835** makro ROC-AUC [cite: 627] skoru elde etmiştir.

## Kurulum ve Çalıştırma

Projeyi çalıştırmak için gerekli kütüphaneler `requirements.txt` dosyasında belirtilmiştir.

```bash
# Gerekli kütüphaneleri kurun
pip install -r requirements.txt
