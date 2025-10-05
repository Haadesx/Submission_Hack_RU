"""
Advanced Real-time Emotion Detection with Recording
Features:
- Multi-face tracking
- Video recording
- Emotion statistics
- Configurable display options
"""

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
from collections import deque, defaultdict
from datetime import datetime
import json
import os

EMOTION_LABELS = {
    0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprised'
}

EMOTION_COLORS = {
    'angry': (0, 0, 255), 'disgusted': (0, 128, 128),
    'fearful': (128, 0, 128), 'happy': (0, 255, 0),
    'neutral': (128, 128, 128), 'sad': (255, 0, 0),
    'surprised': (0, 255, 255)
}


class ImprovedResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ImprovedResNet, self).__init__()
        self.resnet = models.resnet34(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


class ResNetEmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNetEmotionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


class AdvancedEmotionDetector:
    """Advanced detector with recording and statistics"""
    
    def __init__(self, model_path='models/best_advanced_model.pth', device='cuda'):
        print("\n" + "="*60)
        print("  ADVANCED LIVE EMOTION DETECTION")
        print("="*60)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[*] Device: {self.device}")
        
        # Load model
        print(f"[*] Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model_type = checkpoint.get('model_type', 'resnet')
        if 'resnet34' in model_type or 'advanced' in model_type:
            self.model = ImprovedResNet(num_classes=7)
        else:
            self.model = ResNetEmotionModel(num_classes=7)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"[+] Model loaded!")
        
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        print(f"[+] Face detector loaded!")
        
        # Tracking
        self.emotion_history = {}
        self.history_length = 7
        
        # Statistics
        self.emotion_stats = defaultdict(int)
        self.total_frames = 0
        self.faces_detected = 0
        
        # Recording
        self.video_writer = None
        self.recording = False
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("="*60 + "\n")
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces, gray
    
    def predict_emotion(self, face_img):
        pil_img = Image.fromarray(face_img)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        emotion = EMOTION_LABELS[predicted.item()]
        conf = confidence.item()
        probs = probabilities[0].cpu().numpy()
        
        return emotion, conf, probs
    
    def smooth_prediction(self, face_id, emotion, confidence):
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = deque(maxlen=self.history_length)
        
        self.emotion_history[face_id].append((emotion, confidence))
        emotions = [e for e, c in self.emotion_history[face_id]]
        most_common = max(set(emotions), key=emotions.count)
        avg_conf = np.mean([c for e, c in self.emotion_history[face_id] if e == most_common])
        
        return most_common, avg_conf
    
    def draw_ui(self, frame, faces, emotions_detected, fps, show_stats=True):
        """Draw enhanced UI"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for stats
        if show_stats:
            overlay = frame.copy()
            stats_height = 180
            cv2.rectangle(overlay, (0, 0), (300, stats_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Stats text
            y_pos = 25
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25
            cv2.putText(frame, f"Faces: {len(faces)}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25
            cv2.putText(frame, f"Total Frames: {self.total_frames}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25
            
            # Recording indicator
            if self.recording:
                cv2.circle(frame, (20, y_pos - 5), 8, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (35, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_pos += 30
            
            # Top emotions
            if self.emotion_stats:
                cv2.putText(frame, "Top Emotions:", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 20
                
                sorted_emotions = sorted(self.emotion_stats.items(),
                                       key=lambda x: x[1], reverse=True)[:3]
                for emotion, count in sorted_emotions:
                    percentage = (count / max(self.faces_detected, 1)) * 100
                    cv2.putText(frame, f"  {emotion}: {percentage:.1f}%", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                               EMOTION_COLORS[emotion], 1)
                    y_pos += 18
        
        # Controls (bottom left)
        controls = [
            "Q: Quit | S: Stats | R: Record",
            "F: FPS | C: Clear Stats"
        ]
        y_start = h - 50
        for i, text in enumerate(controls):
            cv2.putText(frame, text, (10, y_start + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_emotion_on_face(self, frame, x, y, w, h, emotion, confidence, probs):
        """Draw emotion info on face"""
        color = EMOTION_COLORS[emotion]
        
        # Face rectangle with rounded corners effect
        thickness = 3
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Label with background
        label = f"{emotion}: {confidence*100:.0f}%"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y_label = max(y - 10, label_size[1] + 10)
        
        cv2.rectangle(frame,
                     (x, y_label - label_size[1] - 10),
                     (x + label_size[0] + 10, y_label + baseline),
                     color, -1)
        cv2.putText(frame, label, (x + 5, y_label - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mini probability bars (inside face box)
        if h > 100:  # Only show if face is large enough
            bar_width = w // 7
            bar_x_start = x + 5
            bar_y = y + h - 25
            
            for i, (emo_idx, emo_name) in enumerate(EMOTION_LABELS.items()):
                prob = probs[emo_idx]
                bar_height = int(prob * 20)
                bar_x = bar_x_start + i * (bar_width + 2)
                
                cv2.rectangle(frame,
                             (bar_x, bar_y),
                             (bar_x + bar_width - 2, bar_y - bar_height),
                             EMOTION_COLORS[emo_name], -1)
    
    def start_recording(self, frame_width, frame_height, fps=30.0):
        """Start video recording"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"emotion_detection_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps,
                                           (frame_width, frame_height))
        self.recording = True
        print(f"[*] Recording started: {filename}")
        return filename
    
    def stop_recording(self):
        """Stop video recording"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("[*] Recording stopped")
    
    def save_statistics(self):
        """Save emotion statistics to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = os.path.join(self.output_dir, f"emotion_stats_{timestamp}.json")
        
        stats = {
            'timestamp': timestamp,
            'total_frames': self.total_frames,
            'total_faces_detected': self.faces_detected,
            'emotion_counts': dict(self.emotion_stats),
            'emotion_percentages': {
                emotion: (count / max(self.faces_detected, 1)) * 100
                for emotion, count in self.emotion_stats.items()
            }
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"[*] Statistics saved: {stats_file}")
        return stats_file
    
    def run(self, camera_index=0):
        """Run advanced detection"""
        print("[*] Starting webcam...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("[!] Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("[+] Webcam ready!")
        print("\nControls:")
        print("  Q: Quit | R: Start/Stop Recording")
        print("  S: Toggle Stats | F: Toggle FPS")
        print("  C: Clear Statistics | P: Save Stats")
        print("="*60 + "\n")
        
        fps_list = deque(maxlen=30)
        show_stats = True
        use_smoothing = True
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            self.total_frames += 1
            
            # Detect faces
            faces, gray = self.detect_faces(frame)
            emotions_in_frame = []
            
            # Process each face
            for i, (x, y, w, h) in enumerate(faces):
                self.faces_detected += 1
                face_gray = gray[y:y+h, x:x+w]
                
                emotion, confidence, probs = self.predict_emotion(face_gray)
                
                if use_smoothing:
                    face_id = f"{x//20}_{y//20}"
                    emotion, confidence = self.smooth_prediction(face_id, emotion, confidence)
                
                emotions_in_frame.append(emotion)
                self.emotion_stats[emotion] += 1
                
                self.draw_emotion_on_face(frame, x, y, w, h, emotion, confidence, probs)
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            fps_list.append(fps)
            avg_fps = np.mean(fps_list)
            
            # Draw UI
            frame = self.draw_ui(frame, faces, emotions_in_frame, avg_fps, show_stats)
            
            # Record if enabled
            if self.recording and self.video_writer:
                self.video_writer.write(frame)
            
            # Display
            cv2.imshow('Advanced Emotion Detection', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not self.recording:
                    self.start_recording(frame_width, frame_height, avg_fps)
                else:
                    self.stop_recording()
            elif key == ord('s'):
                show_stats = not show_stats
            elif key == ord('c'):
                self.emotion_stats.clear()
                self.faces_detected = 0
                print("[*] Statistics cleared")
            elif key == ord('p'):
                self.save_statistics()
        
        # Cleanup
        if self.recording:
            self.stop_recording()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n" + "="*60)
        print("  SESSION SUMMARY")
        print("="*60)
        print(f"Total frames processed: {self.total_frames}")
        print(f"Total faces detected: {self.faces_detected}")
        if self.emotion_stats:
            print("\nEmotion Distribution:")
            for emotion, count in sorted(self.emotion_stats.items(),
                                        key=lambda x: x[1], reverse=True):
                percentage = (count / self.faces_detected) * 100
                print(f"  {emotion:12s}: {count:5d} ({percentage:5.1f}%)")
        print("="*60 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Live Emotion Detection')
    parser.add_argument('--model', type=str,
                       default='models/best_advanced_model.pth',
                       help='Path to model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU')
    
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu else 'cuda'
    detector = AdvancedEmotionDetector(model_path=args.model, device=device)
    
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


