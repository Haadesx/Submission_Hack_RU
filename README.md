# Facial Emotion Detection Model

Deep learning model for detecting emotions from facial images using PyTorch.

## Features

- **State-of-the-art CNN Architecture**: Custom improved CNN and ResNet-based models
- **Multiple Dataset Support**: Trained on FER2013 dataset
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs
- **Data Augmentation**: Advanced augmentation techniques for better generalization
- **Progress Tracking**: tqdm integration for visual training progress
- **Class-wise Metrics**: Detailed accuracy metrics for each emotion class

## Emotions Detected

The model can detect 7 different emotions:
- 😠 Angry
- 🤢 Disgusted
- 😨 Fearful
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😮 Surprised

## Dataset

This project uses the **FER2013** (Facial Expression Recognition 2013) dataset:
- **Training images**: ~28,709 images
- **Test images**: ~7,178 images
- **Image size**: 48x48 grayscale
- **Classes**: 7 emotions

Dataset location:
- Train: `D:\RUTGERS\Hackathon\dataset\FER\train`
- Test: `D:\RUTGERS\Hackathon\dataset\FER\test`

## Installation

```bash
pip install -r requirements.txt
```

**Note**: PyTorch with CUDA support should be installed separately based on your system:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Training

### Quick Start

Run the training script with default settings:

```bash
python train_emotion_model.py
```

### Configuration

Edit the configuration in `train_emotion_model.py`:

```python
MODEL_TYPE = 'improved_cnn'  # Options: 'improved_cnn' or 'resnet'
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
```

### Training Features

- **Automatic GPU Detection**: Uses CUDA if available
- **Progress Bars**: tqdm visualization for each epoch
- **Learning Rate Scheduling**: Reduces LR on plateau
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Training History**: JSON log of all metrics
- **Class-wise Accuracy**: Detailed per-class performance

### Expected Training Time

- **GPU (CUDA)**: ~15-20 minutes for 30 epochs
- **CPU**: ~2-3 hours for 30 epochs

## Model Architecture

### Improved CNN (Default)

```
Input (1x48x48) 
  → Conv Block 1 (64 filters)
  → Conv Block 2 (128 filters)
  → Conv Block 3 (256 filters)
  → Conv Block 4 (512 filters)
  → Global Average Pooling
  → FC Layer (512)
  → FC Layer (256)
  → Output (7 classes)
```

**Features**:
- Batch Normalization for stable training
- Dropout for regularization (0.25 for conv, 0.5 for FC)
- Global Average Pooling to reduce parameters
- ReLU activation

### ResNet-based Model

Modified ResNet18 architecture adapted for grayscale emotion detection:
- First conv layer modified for single-channel input
- Custom FC head with dropout
- Transfer learning from ImageNet pretrained weights

## Inference

### Command Line

Predict emotion from a single image:

```bash
python predict_emotion.py path/to/image.jpg
```

Show all emotion probabilities:

```bash
python predict_emotion.py path/to/image.jpg --show-all
```

Use custom model:

```bash
python predict_emotion.py path/to/image.jpg --model models/custom_model.pth
```

### Python API

```python
from predict_emotion import EmotionDetector

# Initialize detector
detector = EmotionDetector(model_path='models/best_emotion_model.pth')

# Predict single image
emotion, confidence, all_probs = detector.predict('path/to/image.jpg')
print(f"Emotion: {emotion}, Confidence: {confidence:.2%}")

# Predict multiple images
results = detector.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
for result in results:
    print(f"{result['image']}: {result['emotion']} ({result['confidence']:.2%})")
```

## Model Performance

Expected validation accuracy: **60-70%** on FER2013

The FER2013 dataset is challenging due to:
- Low resolution (48x48)
- Grayscale images
- Ambiguous labels (inter-rater agreement ~65%)
- Imbalanced classes

### Tips for Better Performance

1. **Increase training epochs**: Try 50-100 epochs
2. **Adjust learning rate**: Fine-tune with lower LR (0.0001)
3. **Use ResNet model**: Better feature extraction
4. **Ensemble models**: Combine multiple model predictions
5. **Data balancing**: Use weighted loss for imbalanced classes

## Project Structure

```
dataset/
├── FER/
│   ├── train/          # Training images (7 emotion folders)
│   └── test/           # Test images (7 emotion folders)
├── archive/            # FEC dataset (Excel files)
├── models/             # Saved models and checkpoints
├── train_emotion_model.py    # Main training script
├── predict_emotion.py        # Inference script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Saved Files

After training, the following files are created:

- `models/best_emotion_model.pth` - Best model checkpoint
- `models/training_history.json` - Training metrics log

Model checkpoint includes:
- Model state dict
- Optimizer state dict
- Training and validation metrics
- Emotion label mapping

## GPU Support

The code automatically detects and uses GPU if available:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Check GPU status:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Troubleshooting

### Out of Memory (OOM) Error

Reduce batch size in training script:
```python
BATCH_SIZE = 32  # or 16
```

### Slow Training

- Enable GPU acceleration
- Reduce `num_workers` in DataLoader
- Use mixed precision training (FP16)

### Low Accuracy

- Increase training epochs
- Add more data augmentation
- Try ResNet model
- Adjust learning rate

## References

- **FER2013 Dataset**: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **FEC Dataset**: [Kaggle](https://www.kaggle.com/datasets/amar09/facial-expression-comparison-fec-google)
- **PyTorch**: [pytorch.org](https://pytorch.org/)

## License

This project is for educational and research purposes.

## Author

Created for Rutgers Hackathon 2025
