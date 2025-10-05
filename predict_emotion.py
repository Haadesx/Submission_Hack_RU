"""
Emotion Detection Inference Script
Load trained model and predict emotions from images
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import os
import numpy as np


class ImprovedEmotionCNN(nn.Module):
    """Enhanced CNN architecture for emotion detection"""
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(ImprovedEmotionCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNetEmotionModel(nn.Module):
    """ResNet-based emotion detection model"""
    def __init__(self, num_classes=7, pretrained=False):
        super(ResNetEmotionModel, self).__init__()
        
        # Load ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)


class EmotionDetector:
    """Emotion detection inference class"""
    
    def __init__(self, model_path='models/best_emotion_model.pth', device=None):
        self.emotion_labels = {
            0: 'angry',
            1: 'disgusted',
            2: 'fearful',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprised'
        }
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load model
        print(f"[*] Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model based on type
        model_type = checkpoint.get('model_type', 'improved_cnn')
        print(f"   Model type: {model_type}")
        
        if model_type == 'resnet':
            self.model = ResNetEmotionModel(num_classes=7)
        else:
            self.model = ImprovedEmotionCNN(num_classes=7)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        val_acc = checkpoint.get('val_acc', 'N/A')
        print(f"   Validation accuracy: {val_acc}")
        print(f"   Device: {self.device}")
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def predict(self, image_path):
        """Predict emotion from image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        emotion = self.emotion_labels[predicted_class]
        
        # Get all probabilities
        all_probs = {self.emotion_labels[i]: probabilities[0][i].item() 
                     for i in range(len(self.emotion_labels))}
        
        return emotion, confidence, all_probs
    
    def predict_batch(self, image_paths):
        """Predict emotions for multiple images"""
        results = []
        for img_path in image_paths:
            emotion, confidence, all_probs = self.predict(img_path)
            results.append({
                'image': img_path,
                'emotion': emotion,
                'confidence': confidence,
                'all_probabilities': all_probs
            })
        return results


def main():
    parser = argparse.ArgumentParser(description='Predict emotion from facial image')
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, default='models/best_emotion_model.pth',
                        help='Path to trained model')
    parser.add_argument('--show-all', action='store_true',
                        help='Show probabilities for all emotions')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"[-] Error: Image not found: {args.image}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"[-] Error: Model not found: {args.model}")
        return
    
    # Create detector
    detector = EmotionDetector(model_path=args.model)
    
    # Predict
    print(f"\n[*] Analyzing image: {args.image}")
    emotion, confidence, all_probs = detector.predict(args.image)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Predicted Emotion: {emotion.upper()}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"{'='*50}")
    
    if args.show_all:
        print(f"\nAll Probabilities:")
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for emotion, prob in sorted_probs:
            bar = '█' * int(prob * 50)
            print(f"  {emotion:12s}: {prob*100:5.2f}% {bar}")
    
    print()


if __name__ == "__main__":
    main()
