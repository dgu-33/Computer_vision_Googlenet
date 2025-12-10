# POC Tissue Classification

GoogleNet-based classification of POC tissue images into 4 classes.

## Dataset

- Training: 4,155 images
- Testing: 1,511 images
- Classes: Chorionic villi, Decidual tissue, Hemorrhage, Trophoblastic tissue

## Requirements

```bash
pip install torch torchvision numpy pillow opencv-python scikit-learn matplotlib seaborn tqdm
```

## Usage

**Required files:** `train_googlenet_final.py`, `poc_dataset.py`, `POC_Dataset/`

```bash
# Training
python train_googlenet_final.py

# Inference
python inference.py
```

## Configuration

- Batch size: 8
- Epochs: 15
- Learning rate: 0.001
- Optimizer: Adam
- Auxiliary loss: Enabled

## Output

- `models/best_googlenet_poc.pth` - Trained model
- `training_history.png` - Training curves
- `confusion_matrix.png` - Test results
