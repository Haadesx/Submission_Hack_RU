"""
HACK TRACKS - AI-Powered Mood-Based Music Therapy Platform

Features:
- Real-time emotion detection from webcam
- Spotify music recommendations based on detected mood
- Gemini AI for intelligent playlist curation and mood insights
- ElevenLabs voice feedback for emotional support
- Music therapy suggestions
- Mood tracking over time
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
import requests
import google.generativeai as genai
from config import (
    GEMINI_API_KEY, GEMINI_MODEL, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI,
    EMOTION_TO_GENRE, EMOTION_TO_MOOD
)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

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
    """ResNet-34 model"""
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
    """ResNet-18 model"""
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


class SpotifyAPI:
    """Spotify API integration"""
    
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.token_expires = 0
    
    def get_token(self):
        """Get Spotify access token"""
        if self.token and time.time() < self.token_expires:
            return self.token
        
        auth_url = 'https://accounts.spotify.com/api/token'
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        try:
            response = requests.post(auth_url, data=auth_data)
            response.raise_for_status()
            data = response.json()
            self.token = data['access_token']
            self.token_expires = time.time() + data['expires_in'] - 60
            return self.token
        except Exception as e:
            print(f"[!] Spotify authentication failed: {e}")
            return None
    
    def search_tracks(self, query, limit=10):
        """Search for tracks"""
        token = self.get_token()
        if not token:
            return []
        
        headers = {'Authorization': f'Bearer {token}'}
        params = {'q': query, 'type': 'track', 'limit': limit}
        
        try:
            response = requests.get('https://api.spotify.com/v1/search',
                                   headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            tracks = []
            for item in data['tracks']['items']:
                tracks.append({
                    'name': item['name'],
                    'artist': item['artists'][0]['name'],
                    'album': item['album']['name'],
                    'url': item['external_urls']['spotify'],
                    'preview_url': item.get('preview_url'),
                    'id': item['id']
                })
            return tracks
        except Exception as e:
            print(f"[!] Spotify search failed: {e}")
            return []
    
    def get_recommendations(self, seed_genres, target_valence=None, 
                          target_energy=None, limit=20):
        """Get track recommendations based on mood"""
        token = self.get_token()
        if not token:
            return []
        
        headers = {'Authorization': f'Bearer {token}'}
        params = {
            'seed_genres': ','.join(seed_genres[:5]),
            'limit': limit
        }
        
        if target_valence is not None:
            params['target_valence'] = target_valence
        if target_energy is not None:
            params['target_energy'] = target_energy
        
        try:
            response = requests.get('https://api.spotify.com/v1/recommendations',
                                   headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            tracks = []
            for item in data['tracks']:
                tracks.append({
                    'name': item['name'],
                    'artist': item['artists'][0]['name'],
                    'album': item['album']['name'],
                    'url': item['external_urls']['spotify'],
                    'preview_url': item.get('preview_url'),
                    'id': item['id']
                })
            return tracks
        except Exception as e:
            print(f"[!] Spotify recommendations failed: {e}")
            return []


class GeminiMoodAnalyzer:
    """Gemini AI for mood analysis and music insights"""
    
    def __init__(self):
        self.model = gemini_model
        self.conversation_history = []
    
    def analyze_emotion_pattern(self, emotion_history):
        """Analyze emotion patterns over time"""
        # Convert deque to list for slicing
        history_list = list(emotion_history)
        emotion_str = ', '.join([f"{e} ({c:.0%})" for e, c in history_list[-10:]])
        
        prompt = f"""As a music therapy AI assistant, analyze this person's emotional state:

Recent emotions detected: {emotion_str}

Provide:
1. A brief emotional assessment (2-3 sentences)
2. Music therapy recommendations
3. Suggested musical characteristics (tempo, energy, mood)
4. Any concerns or positive observations

Keep it supportive and actionable."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Unable to analyze: {e}"
    
    def generate_playlist_description(self, emotion, tracks):
        """Generate a creative playlist description"""
        track_list = '\n'.join([f"- {t['name']} by {t['artist']}" for t in tracks[:5]])
        
        prompt = f"""Create a short, creative playlist description for someone feeling {emotion}.

Top tracks in playlist:
{track_list}

Write 2-3 sentences that are supportive, music-focused, and encouraging."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"A {emotion} playlist to match your mood."
    
    def get_mood_boosting_advice(self, current_emotion, target_emotion='happy'):
        """Get advice for mood improvement"""
        prompt = f"""As a supportive AI music therapist, someone is feeling {current_emotion} 
and wants to feel more {target_emotion}.

Provide:
1. Brief encouragement (1-2 sentences)
2. 3 specific music-based activities to help
3. A motivational closing line

Keep it brief, practical, and positive."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return "Music can be a powerful tool for emotional well-being. Try listening to uplifting songs!"
    
    def suggest_songs_for_emotion(self, emotion):
        """Use Gemini to suggest specific songs for an emotion"""
        prompt = f"""As a music therapy expert, suggest 15 specific real songs (with artist names) 
that would be therapeutic for someone feeling {emotion}.

Requirements:
- Real, popular songs that exist on Spotify
- Mix of genres but appropriate for {emotion} mood
- Include artist name for each
- Format: "Song Title - Artist Name" (one per line)
- No explanations, just the list

Example format:
Someone Like You - Adele
Fix You - Coldplay
etc."""

        try:
            response = self.model.generate_content(prompt)
            # Parse the response into song list
            lines = response.text.strip().split('\n')
            songs = []
            for line in lines:
                line = line.strip()
                # Remove numbering like "1. " or "- "
                line = line.lstrip('0123456789.- ')
                if ' - ' in line and len(line) > 5:
                    songs.append(line)
            return songs[:15]  # Limit to 15
        except Exception as e:
            print(f"[!] Gemini song suggestion failed: {e}")
            return []
    
    def create_personalized_message(self, emotion, name="Friend"):
        """Create a personalized voice message"""
        prompt = f"""Create a brief, warm message for someone who is feeling {emotion}.

Address them as "{name}" and:
1. Acknowledge their emotion (1 sentence)
2. Offer brief encouragement (1 sentence)
3. Suggest they enjoy the music (1 sentence)

Keep it natural and conversational, under 50 words total."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Hi {name}! I've curated some music for your current mood. Enjoy!"


class ElevenLabsVoice:
    """ElevenLabs text-to-speech integration"""
    
    def __init__(self, api_key, voice_id):
        self.api_key = api_key
        self.voice_id = voice_id
        self.url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    def speak(self, text, save_path="output.mp3"):
        """Convert text to speech"""
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        try:
            response = requests.post(self.url, json=data, headers=headers)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            print(f"[+] Voice message saved: {save_path}")
            return save_path
        except Exception as e:
            print(f"[!] ElevenLabs TTS failed: {e}")
            return None


class HackTracksSystem:
    """Main Hack Tracks system"""
    
    def __init__(self, model_path='models/best_advanced_model.pth'):
        print("\n" + "="*60)
        print("  [HACK TRACKS] - Mood-Based Music Therapy")
        print("="*60)
        
        # Load emotion detection model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[*] Device: {self.device}")
        
        print(f"[*] Loading emotion model...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model_type = checkpoint.get('model_type', 'resnet')
        if 'resnet34' in model_type or 'advanced' in model_type:
            self.emotion_model = ImprovedResNet(num_classes=7)
        else:
            self.emotion_model = ResNetEmotionModel(num_classes=7)
        
        self.emotion_model.load_state_dict(checkpoint['model_state_dict'])
        self.emotion_model.to(self.device)
        self.emotion_model.eval()
        print(f"[+] Emotion model loaded!")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        print(f"[+] Face detector loaded!")
        
        # Initialize APIs
        print(f"[*] Initializing Gemini AI...")
        self.gemini = GeminiMoodAnalyzer()
        print(f"[+] Gemini AI ready!")
        
        print(f"[*] Initializing ElevenLabs Voice...")
        self.voice = ElevenLabsVoice(ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID)
        print(f"[+] ElevenLabs ready!")
        
        if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
            print(f"[*] Initializing Spotify API...")
            self.spotify = SpotifyAPI(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
            print(f"[+] Spotify API ready!")
        else:
            print(f"[!] Spotify API not configured (set keys in config.py)")
            self.spotify = None
        
        # Tracking
        self.emotion_history = deque(maxlen=100)
        self.emotion_counts = defaultdict(int)
        self.current_emotion = None
        self.current_confidence = 0.0
        self.last_recommendation_time = 0
        self.recommendation_cooldown = 30  # seconds
        
        # Output
        os.makedirs('playlists', exist_ok=True)
        os.makedirs('voice_messages', exist_ok=True)
        
        print("="*60 + "\n")
    
    def detect_emotion(self, face_img):
        """Detect emotion from face"""
        pil_img = Image.fromarray(face_img)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.emotion_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        emotion = EMOTION_LABELS[predicted.item()]
        conf = confidence.item()
        probs = probabilities[0].cpu().numpy()
        
        return emotion, conf, probs
    
    def get_music_recommendations(self, emotion):
        """Get music recommendations for emotion using Gemini AI + Spotify"""
        if not self.spotify:
            print("[!] Spotify not configured")
            return []
        
        print(f"[*] Getting recommendations for {emotion} mood...")
        
        # Step 1: Ask Gemini AI to suggest specific songs
        print(f"[*] Asking Gemini AI for song suggestions...")
        gemini_songs = self.gemini.suggest_songs_for_emotion(emotion)
        
        if gemini_songs:
            print(f"[+] Gemini suggested {len(gemini_songs)} songs")
            
            # Step 2: Search Spotify for each suggested song
            tracks = []
            for song in gemini_songs[:10]:  # Limit to 10 to avoid too many API calls
                search_results = self.spotify.search_tracks(song, limit=1)
                if search_results:
                    tracks.extend(search_results)
            
            if tracks:
                print(f"[+] Found {len(tracks)} songs on Spotify")
                return tracks
        
        # Fallback: If Gemini fails, use mood-based search
        print(f"[*] Using mood-based search fallback...")
        moods = EMOTION_TO_MOOD.get(emotion, ['music'])
        search_query = f"{emotion} {moods[0]} therapy"
        tracks = self.spotify.search_tracks(search_query, limit=15)
        
        return tracks
    
    def save_playlist(self, emotion, tracks):
        """Save playlist to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"playlists/{emotion}_playlist_{timestamp}.json"
        
        # Get Gemini description
        description = self.gemini.generate_playlist_description(emotion, tracks)
        
        playlist_data = {
            'timestamp': timestamp,
            'emotion': emotion,
            'description': description,
            'tracks': tracks
        }
        
        with open(filename, 'w') as f:
            json.dump(playlist_data, f, indent=4)
        
        print(f"[+] Playlist saved: {filename}")
        return filename, description
    
    def generate_voice_message(self, emotion):
        """Generate personalized voice message with therapy guidance"""
        # Get enhanced message from Gemini
        message = self.gemini.create_personalized_message(emotion)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"voice_messages/{emotion}_{timestamp}.mp3"
        
        # Generate voice using ElevenLabs
        self.voice.speak(message, filename)
        return filename, message
    
    def generate_therapy_session(self, emotion):
        """Generate a full therapy session narration"""
        prompt = f"""Create a 2-3 minute calming therapy session script for someone feeling {emotion}.

Include:
1. Warm greeting and acknowledgment of their emotion
2. Breathing exercise guidance
3. Positive affirmations
4. Music therapy benefits
5. Encouraging closing

Write it as a script to be read aloud, natural and conversational."""
        
        try:
            response = self.gemini.model.generate_content(prompt)
            script = response.text
            
            # Generate longer voice message
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"voice_messages/therapy_{emotion}_{timestamp}.mp3"
            self.voice.speak(script, filename)
            
            return filename, script
        except Exception as e:
            print(f"[!] Therapy session generation failed: {e}")
            return None, None
    
    def run_interactive(self, camera_index=0):
        """Run interactive mood detection and music recommendation"""
        print("[*] Starting webcam...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("[!] Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("[+] Webcam ready!")
        print("\nControls:")
        print("  Q: Quit")
        print("  R: Get music recommendations (Gemini + Spotify)")
        print("  V: Generate voice message")
        print("  T: Full therapy session (Gemini + ElevenLabs)")
        print("  A: Get mood analysis")
        print("  S: Save current playlist")
        print("="*60 + "\n")
        
        fps_list = deque(maxlen=30)
        current_tracks = []
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Process first face
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_gray = gray[y:y+h, x:x+w]
                
                emotion, confidence, probs = self.detect_emotion(face_gray)
                
                self.current_emotion = emotion
                self.current_confidence = confidence
                self.emotion_history.append((emotion, confidence))
                self.emotion_counts[emotion] += 1
                
                # Draw results
                color = EMOTION_COLORS[emotion]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                label = f"{emotion}: {confidence*100:.0f}%"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw UI
            h, w_frame = frame.shape[:2]
            
            # Stats overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (400, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # FPS
            fps = 1.0 / (time.time() - start_time)
            fps_list.append(fps)
            avg_fps = np.mean(fps_list)
            
            y_pos = 30
            cv2.putText(frame, f"HACK TRACKS", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_pos += 35
            cv2.putText(frame, f"Mood: {self.current_emotion or 'Detecting...'}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 30
            cv2.putText(frame, f"Confidence: {self.current_confidence*100:.0f}%", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 30
            
            # Top emotion
            if self.emotion_counts:
                top_emotion = max(self.emotion_counts, key=self.emotion_counts.get)
                cv2.putText(frame, f"Dominant: {top_emotion}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           EMOTION_COLORS[top_emotion], 2)
            
            # Controls
            controls = [
                "Q: Quit | R: Recommend | T: Therapy",
                "V: Voice | A: Analysis | S: Save"
            ]
            y_start = h - 60
            for i, text in enumerate(controls):
                cv2.putText(frame, text, (10, y_start + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Hack Tracks - Mood Detection', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('r'):
                if self.current_emotion:
                    print(f"\n[*] Getting recommendations for {self.current_emotion} mood...")
                    current_tracks = self.get_music_recommendations(self.current_emotion)
                    
                    if current_tracks:
                        print(f"\n{'='*60}")
                        print(f"[{self.current_emotion.upper()} MOOD PLAYLIST]")
                        print(f"{'='*60}")
                        for i, track in enumerate(current_tracks[:10], 1):
                            print(f"{i:2d}. {track['name']} - {track['artist']}")
                            if track['url']:
                                print(f"    Link: {track['url']}")
                        print(f"{'='*60}\n")
                    else:
                        print("[!] No tracks found")
            
            elif key == ord('v'):
                if self.current_emotion:
                    print(f"\n[*] Generating voice message for {self.current_emotion} mood...")
                    filename, message = self.generate_voice_message(self.current_emotion)
                    print(f"\n[Message]: {message}")
                    print(f"[Audio saved to]: {filename}\n")
            
            elif key == ord('t'):
                if self.current_emotion:
                    print(f"\n[*] Creating full therapy session for {self.current_emotion} mood...")
                    print("[*] This may take a minute (Gemini + ElevenLabs)...")
                    filename, script = self.generate_therapy_session(self.current_emotion)
                    if filename:
                        print(f"\n{'='*60}")
                        print("[THERAPY SESSION GENERATED]")
                        print(f"{'='*60}")
                        print(script[:200] + "..." if len(script) > 200 else script)
                        print(f"\n[Audio saved to]: {filename}")
                        print(f"[Duration]: ~2-3 minutes")
                        print(f"{'='*60}\n")
            
            elif key == ord('a'):
                if len(self.emotion_history) > 5:
                    print(f"\n[*] Analyzing emotional patterns...")
                    analysis = self.gemini.analyze_emotion_pattern(self.emotion_history)
                    print(f"\n{'='*60}")
                    print("[MOOD ANALYSIS]")
                    print(f"{'='*60}")
                    print(analysis)
                    print(f"{'='*60}\n")
            
            elif key == ord('s'):
                if current_tracks:
                    filename, desc = self.save_playlist(self.current_emotion, current_tracks)
                    print(f"\n[Playlist Description]:")
                    print(desc)
                    print()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final summary
        print("\n" + "="*60)
        print("  SESSION SUMMARY")
        print("="*60)
        print(f"Frames processed: {len(self.emotion_history)}")
        if self.emotion_counts:
            print("\nEmotion Distribution:")
            total = sum(self.emotion_counts.values())
            for emotion, count in sorted(self.emotion_counts.items(),
                                        key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                print(f"  {emotion:12s}: {percentage:5.1f}%")
        print("="*60 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hack Tracks - Mood-Based Music Therapy')
    parser.add_argument('--model', type=str,
                       default='models/best_advanced_model.pth',
                       help='Emotion detection model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index')
    
    args = parser.parse_args()
    
    system = HackTracksSystem(model_path=args.model)
    
    try:
        system.run_interactive(camera_index=args.camera)
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
