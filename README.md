# Ship Image Classification using EfficientNetB0

# Ship Image Classification using EfficientNet-B0

## Overview
This project performs **ship image classification** into five categories  
(**Cargo, Carrier, Cruise, Military, Tanker**) using **deep learning and transfer learning**.  
The model is based on **EfficientNet-B0** pretrained on ImageNet and fine-tuned using PyTorch.

Both **training without data augmentation (NO-AUG)** and **training with data augmentation (AUG)** are implemented and compared.

## Dataset
**Kaggle – Game of Deep Learning: Ship Dataset**

---

## Model
- **Architecture:** EfficientNet-B0  
- **Pretrained:** ImageNet  
- **Input size:** 224 × 224  
- **Loss function:** CrossEntropyLoss  
- **Optimizer:** Adam (lr = 3e-4)  
- **Batch size:** 16  
- **Epochs:** 10  
- **LR Scheduler:** ReduceLROnPlateau  

---

## Training Strategy
Two experiments are conducted:

1. **NO-AUG**  
   - Standard normalization only  
2. **AUG**  
   - Random horizontal flip  
   - Random rotation  
   - Color jitter  

The best model is selected based on **validation accuracy**.

---

## Results
| Experiment | Best Validation Accuracy |
|----------|--------------------------|
| NO-AUG   | ≈ **0.955** |
| AUG      | ≈ **0.953** |

Additional outputs:
- Classification report (`.txt`)
- Metrics (`.json`)
- Confusion matrix (`.png`)
- Best model weights (`.pth`)

All results are saved under the `outputs/` directory.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt


## License

MIT Lisansı © 2025 Celil Berk Çoban
