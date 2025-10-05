# 🎵 HACK TRACKS
### AI-Powered Mood-Based Music Therapy Platform

> *Detect emotions in real-time and get personalized music recommendations*

![Hack Tracks Banner](https://img.shields.io/badge/RU%20Hacks-2025-blue) 
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🎯 **What is Hack Tracks?**

**Hack Tracks** is an innovative music therapy platform that combines:
- 🎭 **AI Emotion Detection** - Detects 7 emotions from your face in real-time
- 🎵 **Smart Music Recommendations** - Spotify playlists matched to your mood
- 🧠 **AI Mood Analysis** - Gemini AI provides emotional insights and support
- 🎤 **Voice Encouragement** - ElevenLabs generates personalized motivational messages
- 📊 **Mood Tracking** - Monitor your emotional well-being over time

**Demo in 3 steps:**
1. Face webcam → Emotion detected
2. Press 'R' → Get 15 mood-matched songs
3. Press 'V' → Hear AI encouragement

---

## 🏆 **Hackathon Categories**

### **Primary:**
- 🌟 **Social Good** - Mental health through accessible music therapy

### **Superlatives:**
- ✅ **Best Use of Gemini API** (MLH) - Mood analysis, therapy insights, playlist curation
- ✅ **Best Use of ElevenLabs** (MLH) - Voice synthesis for emotional support
- ✅ **Best UI/UX Design** - Real-time visualization, intuitive controls
- ✅ **Best Entrepreneurial Hack** (IDEA) - Viable mental health SaaS model

---

## ✨ **Key Features**

### **1. Real-Time Emotion Detection** 🎭
- ResNet-34 deep learning model
- 61.28% accuracy (matches research!)
- 7 emotions: happy, sad, angry, fearful, surprised, disgusted, neutral
- 25-30 FPS on GPU
- Multi-face support

### **2. Spotify Music Recommendations** 🎵
- Mood-matched playlists
- Genre and energy mapping
- 15+ curated tracks per emotion
- Direct Spotify links

### **3. Gemini AI Mood Analysis** 🧠
- Emotional pattern recognition
- Music therapy suggestions
- Personalized insights
- Mood-boosting strategies

### **4. ElevenLabs Voice Support** 🎤
- Natural AI voice messages
- Personalized encouragement
- Emotion-specific support
- MP3 exports

### **5. Analytics & Tracking** 📊
- Emotion history
- Dominant mood detection
- Session summaries
- JSON export

---

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Set Up Spotify API** (Required)

Get credentials from [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)

Add to `config.py`:
```python
SPOTIFY_CLIENT_ID = "your_client_id"
SPOTIFY_CLIENT_SECRET = "your_client_secret"
```

[📖 Detailed Spotify Setup Guide →](SPOTIFY_SETUP.md)

### **3. Run Hack Tracks**
```bash
python hack_tracks.py
```

### **4. Use It!**
- **R** - Get music recommendations
- **V** - Generate voice message
- **A** - Get AI mood analysis
- **S** - Save playlist
- **Q** - Quit

---

## 🎮 **How It Works**

```
┌─────────────┐
│  Webcam     │
│  Face Image │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ ResNet-34 Model │  ← Emotion Detection (61% accuracy)
│ Emotion: Happy  │
└──────┬──────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│              Hack Tracks System              │
├──────────────────────────────────────────────┤
│                                              │
│  ┌─────────────┐  ┌──────────────┐         │
│  │ Spotify API │  │  Gemini AI   │         │
│  │ Playlists   │  │  Analysis    │         │
│  └─────────────┘  └──────────────┘         │
│                                              │
│  ┌─────────────┐  ┌──────────────┐         │
│  │ ElevenLabs  │  │  Analytics   │         │
│  │ Voice       │  │  Tracking    │         │
│  └─────────────┘  └──────────────┘         │
│                                              │
└──────────────────────────────────────────────┘
       │
       ▼
┌──────────────────┐
│  Music + Voice   │  ← Personalized therapy
│  Insights        │
└──────────────────┘
```

---

## 📊 **Performance**

### **Emotion Detection:**
| Emotion | Accuracy |
|---------|----------|
| Happy | 78.97% ✅ |
| Surprised | 78.94% ✅ |
| Neutral | 67.88% ✅ |
| Angry | 60.23% ✅ |
| Overall | **61.28%** |

### **System:**
- **FPS:** 25-30 on RTX 3060
- **Latency:** ~35ms per frame
- **API Response:** <2s
- **Voice Gen:** ~3s per message

---

## 💡 **Tech Stack**

```
Frontend:   OpenCV (real-time video processing)
ML Model:   PyTorch + ResNet-34
Training:   FER2013 dataset (35,887 images)
AI APIs:    
  - Google Gemini 2.0 Flash (mood analysis)
  - ElevenLabs API (voice synthesis)
  - Spotify Web API (music recommendations)
Language:   Python 3.10+
Hardware:   CUDA-enabled GPU (optional, runs on CPU)
```

---

## 📁 **Project Structure**

```
hack-tracks/
├── 🎵 hack_tracks.py              # Main application
├── 🎥 live_emotion_detection.py   # Standalone detector
├── 🎥 live_emotion_advanced.py    # Advanced detector
│
├── 🤖 models/
│   ├── best_advanced_model.pth    # ResNet-34 (61.28%)
│   ├── best_balanced_model.pth    # ResNet-18 (60.25%)
│   └── best_emotion_model.pth     # ResNet-18 (58%)
│
├── 🎼 Training Scripts
│   ├── train_emotion_model.py
│   ├── train_balanced_model.py
│   └── train_advanced_model.py
│
├── 📊 FER Dataset
│   ├── train/ (28,709 images)
│   └── test/ (7,178 images)
│
├── 📚 Documentation
│   ├── README_HACK_TRACKS.md      # This file
│   ├── HACKATHON_GUIDE.md         # Comprehensive guide
│   ├── SPOTIFY_SETUP.md           # API setup
│   └── LIVE_DETECTION_GUIDE.md    # Usage guide
│
├── ⚙️ Configuration
│   ├── config.py                  # API keys & settings
│   └── requirements.txt           # Dependencies
│
└── 📁 Output
    ├── playlists/                 # Saved playlists
    └── voice_messages/            # AI voice files
```

---

## 🎯 **Use Cases**

### **1. Personal Wellness** 💆
- Daily mood check-ins
- Music therapy at home
- Stress management
- Emotional awareness

### **2. Clinical Settings** 🏥
- Therapist support tool
- Patient progress tracking
- Music prescription
- Supplement therapy

### **3. Workplace Wellness** 💼
- Employee well-being
- Stress monitoring
- Break room tool
- Team morale

### **4. Education** 🎓
- Student mental health
- Counseling support
- Wellness programs
- Emotional literacy

---

## 🌟 **Why It Matters**

### **The Problem:**
- 1 in 5 adults face mental health challenges
- Music therapy works but is expensive ($100-200/session)
- Limited access to therapists
- Stigma around seeking help

### **Our Solution:**
- ✅ **Free & Accessible** - No appointments, no stigma
- ✅ **Immediate** - Results in seconds
- ✅ **Private** - Runs locally, no data uploaded
- ✅ **Evidence-Based** - Music therapy is clinically proven
- ✅ **Scalable** - Can reach millions

### **Impact:**
- 🎯 **Democratize** music therapy
- 🎯 **Reduce** mental health burden
- 🎯 **Increase** emotional awareness
- 🎯 **Improve** accessibility

---

## 🎨 **Screenshots & Demo**

### **Main Interface:**
```
┌─────────────────────────────────────────┐
│  HACK TRACKS                            │
│  Mood: Happy                            │
│  Confidence: 87%                        │
│  FPS: 28.5                              │
│  Dominant: Happy                        │
├─────────────────────────────────────────┤
│                                         │
│     [Your face with green box]          │
│     Happy: 87%                          │
│                                         │
│     [Emotion probability bars]          │
│                                         │
├─────────────────────────────────────────┤
│  Q: Quit | R: Recommend                 │
│  V: Voice | A: Analysis | S: Save       │
└─────────────────────────────────────────┘
```

### **Music Recommendations:**
```
============================================================
🎵 HAPPY MOOD PLAYLIST 🎵
============================================================
 1. Good 4 U - Olivia Rodrigo
    🔗 https://open.spotify.com/track/...
 2. Levitating - Dua Lipa
    🔗 https://open.spotify.com/track/...
 3. Blinding Lights - The Weeknd
    🔗 https://open.spotify.com/track/...
...
============================================================
```

---

## 💰 **Business Model**

### **Revenue Streams:**
1. **Freemium** ($0-9/mo) - Free basic, Pro unlimited
2. **B2B Enterprise** ($500-5000/mo) - Corporate wellness
3. **Clinical** ($1000-10000/mo) - Healthcare providers
4. **API Access** ($0.01-0.10/request) - Developer platform

### **Market:**
- Mental Health Apps: $5.2B
- Music Streaming: $38.7B  
- Music Therapy: $2.1B
- **TAM:** ~$10B

---

## 🚧 **Future Roadmap**

### **Short-term (3 months):**
- ✅ Mobile app (React Native)
- ✅ More music services (Apple Music, YouTube)
- ✅ Playlist sharing
- ✅ Social features

### **Medium-term (6-12 months):**
- ✅ Voice control
- ✅ Multi-user support
- ✅ Music generation (ElevenLabs Music)
- ✅ Therapist dashboard

### **Long-term (1-2 years):**
- ✅ Clinical trials
- ✅ Insurance integration
- ✅ Smart home
- ✅ Wearable sync

---

## 📚 **Documentation**

- [**Hackathon Guide**](HACKATHON_GUIDE.md) - Comprehensive submission guide
- [**Spotify Setup**](SPOTIFY_SETUP.md) - API configuration
- [**Usage Guide**](LIVE_DETECTION_GUIDE.md) - How to use
- [**Project Complete**](PROJECT_COMPLETE.md) - Technical details

---

## 🤝 **Contributing**

This is a hackathon project, but contributions are welcome!

1. Fork the repo
2. Create feature branch
3. Make changes
4. Submit pull request

---

## 📄 **License**

MIT License - See LICENSE file

---

## 👥 **Team**

Built with ❤️ for RU Hacks 2025

**Tech Stack:**
- PyTorch, OpenCV, Python
- Google Gemini API
- ElevenLabs API
- Spotify Web API

---

## 🙏 **Acknowledgments**

- **FER2013 Dataset** - Emotion recognition training
- **Google Gemini** - AI-powered mood analysis
- **ElevenLabs** - Natural voice synthesis
- **Spotify** - Music recommendations
- **MLH** - Hackathon organization

---

## 📞 **Contact & Links**

- **Demo:** [Video Link]
- **Pitch:** [Slides Link]
- **DevPost:** [Project Link]
- **GitHub:** [Repo Link]

---

## ⭐ **Show Your Support**

If you like this project, please ⭐ star the repo!

---

<div align="center">

### 🎵 **Music has the power to heal. AI has the power to understand.** 🎵

**Together, they create Hack Tracks.**

[Get Started](#-quick-start) • [Documentation](#-documentation) • [Demo](# )

</div>

---

*"Where words fail, music speaks." - Hans Christian Andersen*
