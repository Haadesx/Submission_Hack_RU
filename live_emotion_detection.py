"""
Real-time Emotion Detection using OpenCV
Detects faces in live video and classifies emotions
"""

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
from collections import deque

# Emotion labels
EMOTION_LABELS = {
    0: 'angry',
    1: 'disgusted', 
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised'
}

# Colors for each emotion (BGR format for OpenCV)
EMOTION_COLORS = {
    'angry': (0, 0, 255),        # Red
    'disgusted': (0, 128, 128),  # Teal
    'fearful': (128, 0, 128),    # Purple
    'happy': (0, 255, 0),        # Green
    'neutral': (128, 128, 128),  # Gray
    'sad': (255, 0, 0),          # Blue
    'surprised': (0, 255, 255)   # Yellow
}


class ImprovedResNet(nn.Module):
    """ResNet-34 model architecture"""
    def __init__(self, num_classes=7):
        super(ImprovedResNet, self).__init__()
        self.resnet = models.resnet34(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


class ResNetEmotionModel(nn.Module):
    """ResNet-18 model architecture (for balanced model)"""
    def __init__(self, num_classes=7):
        super(ResNetEmotionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


class LiveEmotionDetector:
    """Real-time emotion detection from webcam"""
    
    def __init__(self, model_path='models/best_advanced_model.pth', device='cuda'):
        print("\n" + "="*60)
        print("  LIVE EMOTION DETECTION - Initializing...")
        print("="*60)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[*] Using device: {self.device}")
        
        # Load model
        print(f"[*] Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model type and initialize
        model_type = checkpoint.get('model_type', 'resnet')
        if 'resnet34' in model_type or 'advanced' in model_type:
            self.model = ImprovedResNet(num_classes=7)
        else:
            self.model = ResNetEmotionModel(num_classes=7)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"[+] Model loaded successfully!")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load face detector (Haar Cascade - fast and reliable)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        print(f"[+] Face detector loaded!")
        
        # For smoothing predictions
        self.emotion_history = {}
        self.history_length = 5  # Average last 5 predictions
        
        print("="*60 + "\n")
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces, gray
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image"""
        # Convert to PIL Image
        pil_img = Image.fromarray(face_img)
        
        # Preprocess
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        emotion = EMOTION_LABELS[predicted.item()]
        conf = confidence.item()
        probs = probabilities[0].cpu().numpy()
        
        return emotion, conf, probs
    
    def smooth_prediction(self, face_id, emotion, confidence):
        """Smooth predictions using history"""
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = deque(maxlen=self.history_length)
        
        self.emotion_history[face_id].append((emotion, confidence))
        
        # Vote for most common emotion
        emotions = [e for e, c in self.emotion_history[face_id]]
        most_common = max(set(emotions), key=emotions.count)
        avg_conf = np.mean([c for e, c in self.emotion_history[face_id] if e == most_common])
        
        return most_common, avg_conf
    
    def draw_emotion(self, frame, x, y, w, h, emotion, confidence, probs):
        """Draw bounding box and emotion label"""
        color = EMOTION_COLORS[emotion]
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Prepare label
        label = f"{emotion}: {confidence*100:.1f}%"
        
        # Draw label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_label = max(y - 10, label_size[1] + 10)
        cv2.rectangle(frame, 
                     (x, y_label - label_size[1] - 10),
                     (x + label_size[0], y_label + baseline),
                     color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x, y_label - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw emotion bar chart (top emotions)
        bar_height = 15
        bar_y_start = y + h + 10
        
        # Get top 3 emotions
        top_indices = np.argsort(probs)[-3:][::-1]
        for i, idx in enumerate(top_indices):
            emotion_name = EMOTION_LABELS[idx]
            prob = probs[idx]
            
            bar_width = int(prob * 150)
            bar_y = bar_y_start + i * (bar_height + 5)
            
            # Ensure bars don't go off screen
            if bar_y + bar_height > frame.shape[0]:
                break
            
            # Draw bar
            cv2.rectangle(frame,
                         (x, bar_y),
                         (x + bar_width, bar_y + bar_height),
                         EMOTION_COLORS[emotion_name], -1)
            
            # Draw emotion name
            cv2.putText(frame, f"{emotion_name}", 
                       (x + bar_width + 5, bar_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self, camera_index=0, show_fps=True):
        """Run live detection"""
        print("[*] Starting webcam...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("[!] Error: Could not open webcam")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("[+] Webcam opened successfully!")
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to toggle smoothing")
        print("  - Press 'f' to toggle FPS display")
        print("="*60 + "\n")
        
        fps_list = deque(maxlen=30)
        use_smoothing = True
        show_fps_flag = show_fps
        
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("[!] Error: Failed to capture frame")
                break
            
            # Detect faces
            faces, gray = self.detect_faces(frame)
            
            # Process each face
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face region
                face_gray = gray[y:y+h, x:x+w]
                
                # Predict emotion
                emotion, confidence, probs = self.predict_emotion(face_gray)
                
                # Smooth prediction
                if use_smoothing:
                    face_id = f"{x}_{y}"  # Simple face tracking
                    emotion, confidence = self.smooth_prediction(face_id, emotion, confidence)
                
                # Draw results
                self.draw_emotion(frame, x, y, w, h, emotion, confidence, probs)
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            fps_list.append(fps)
            avg_fps = np.mean(fps_list)
            
            # Draw info
            info_text = []
            if show_fps_flag:
                info_text.append(f"FPS: {avg_fps:.1f}")
            info_text.append(f"Faces: {len(faces)}")
            info_text.append(f"Smoothing: {'ON' if use_smoothing else 'OFF'}")
            
            y_offset = 30
            for text in info_text:
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # Display frame
            cv2.imshow('Live Emotion Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                use_smoothing = not use_smoothing
                print(f"[*] Smoothing: {'ON' if use_smoothing else 'OFF'}")
            elif key == ord('f'):
                show_fps_flag = not show_fps_flag
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n[+] Webcam closed")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Emotion Detection')
    parser.add_argument('--model', type=str, 
                       default='models/best_advanced_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Initialize detector
    device = 'cpu' if args.cpu else 'cuda'
    detector = LiveEmotionDetector(model_path=args.model, device=device)
    
    # Run detection
    try:
        detector.run(camera_index=args.camera)
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


