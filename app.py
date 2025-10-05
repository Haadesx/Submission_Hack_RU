"""
HACK TRACKS - Multi-Page Web Application
Professional emotion-based music therapy platform
"""

import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
import google.generativeai as genai
import requests
from io import BytesIO
import os
import random
from datetime import datetime

from config import (
    GEMINI_API_KEY, GEMINI_MODEL, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    ELEVENLABS_API_URL, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, 
    EMOTION_TO_GENRE, EMOTION_TO_MOOD, BAD_MOODS
)
from hack_tracks import (
    ImprovedResNet, ResNetEmotionModel, SpotifyAPI, 
    GeminiMoodAnalyzer, ElevenLabsVoice, EMOTION_LABELS
)

# Page config
st.set_page_config(
    page_title="Hack Tracks - AI Wellness Platform",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with uniform design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 900;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: rgba(255,255,255,0.95);
        font-weight: 500;
    }
    
    /* Card System */
    .feature-card {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 2px solid transparent;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.25);
        border-color: #667eea;
    }
    
    .card-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1rem;
    }
    
    .card-text {
        font-size: 1.1rem;
        color: #4a5568;
        line-height: 1.7;
    }
    
    /* Emotion Display */
    .emotion-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 25px 70px rgba(102, 126, 234, 0.4);
        margin: 2rem 0;
    }
    
    .emotion-label {
        font-size: 4.5rem;
        font-weight: 900;
        color: white;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }
    
    .emotion-confidence {
        font-size: 2rem;
        color: rgba(255,255,255,0.95);
        font-weight: 600;
        margin-top: 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1.2rem 2rem;
        font-size: 1.15rem;
        border-radius: 15px;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.35);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Sidebar Navigation */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Alert Boxes */
    .alert {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        color: #0c4a6e;
        border-left: 5px solid #0284c7;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        border-left: 5px solid #10b981;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #78350f;
        border-left: 5px solid #f59e0b;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 12px;
        border-radius: 10px;
    }
    
    /* Track Cards */
    .track-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 18px;
        margin: 1rem 0;
        border-left: 6px solid #667eea;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .track-card:hover {
        transform: translateX(8px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        border-left-color: #764ba2;
    }
    
    /* Navigation Pills */
    .nav-pill {
        display: inline-block;
        padding: 0.8rem 2rem;
        margin: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50px;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .nav-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Section Headers */
    .section-header {
        text-align: center;
        margin: 4rem 0 2rem 0;
    }
    
    .section-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .section-subtitle {
        font-size: 1.3rem;
        color: #4a5568;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'emotion_model' not in st.session_state:
    st.session_state.emotion_model = None
if 'spotify' not in st.session_state:
    st.session_state.spotify = None
if 'gemini' not in st.session_state:
    st.session_state.gemini = None
if 'voice' not in st.session_state:
    st.session_state.voice = None
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = None
if 'current_tracks' not in st.session_state:
    st.session_state.current_tracks = []
if 'location' not in st.session_state:
    st.session_state.location = None
if 'recommendation_seed' not in st.session_state:
    st.session_state.recommendation_seed = random.randint(0, 1000000)

# Load resources
@st.cache_resource
def load_emotion_model():
    """Load emotion detection model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('models/best_advanced_model.pth', map_location=device)
    model_type = checkpoint.get('model_type', 'resnet')
    
    if 'resnet34' in model_type or 'advanced' in model_type:
        model = ImprovedResNet(num_classes=7)
    else:
        model = ResNetEmotionModel(num_classes=7)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_face_detector():
    """Load face detector"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

@st.cache_resource
def initialize_apis():
    """Initialize all APIs"""
    spotify = SpotifyAPI(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    gemini = GeminiMoodAnalyzer()
    voice = ElevenLabsVoice(ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID)
    return spotify, gemini, voice

def get_user_location():
    """Get user's location from IP address"""
    try:
        response = requests.get('https://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'country': data.get('country_name', 'Unknown'),
                'timezone': data.get('timezone', 'Unknown'),
                'latitude': data.get('latitude'),
                'longitude': data.get('longitude')
            }
    except:
        pass
    return None

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="hero-section" style="padding: 2rem 1rem; margin-bottom: 2rem;"><h1 style="color: white; font-size: 2.5rem; margin: 0;">🎵 HACK TRACKS</h1><p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem;">AI Wellness Platform</p></div>', unsafe_allow_html=True)
    
    st.markdown("### 🧭 Navigation")
    
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = 'home'
        st.rerun()
    
    if st.button("🎭 Detect Emotion", use_container_width=True):
        st.session_state.page = 'detect'
        st.rerun()
    
    if st.button("🎵 Music Therapy", use_container_width=True):
        st.session_state.page = 'music'
        st.rerun()
    
    if st.button("🤖 AI Therapy", use_container_width=True):
        st.session_state.page = 'therapy'
        st.rerun()
    
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.page = 'about'
        st.rerun()
    
    st.markdown("---")
    
    # Location display
    if st.session_state.location:
        st.markdown(f"""
        <div class="alert alert-info" style="font-size: 0.9rem; padding: 1rem;">
            📍 {st.session_state.location['city']}, {st.session_state.location['country']}
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    if st.session_state.current_emotion:
        st.markdown(f"""
        <div class="alert alert-success" style="font-size: 0.9rem; padding: 1rem;">
            <strong>Current Mood:</strong><br/>
            {st.session_state.current_emotion.upper()}
        </div>
        """, unsafe_allow_html=True)

# Initialize resources
if st.session_state.emotion_model is None:
    with st.spinner("🔄 Loading AI models..."):
        st.session_state.emotion_model, st.session_state.device = load_emotion_model()
        st.session_state.face_cascade = load_face_detector()
        st.session_state.spotify, st.session_state.gemini, st.session_state.voice = initialize_apis()

if st.session_state.location is None:
    st.session_state.location = get_user_location()

# Page Router
if st.session_state.page == 'home':
    exec(open('pages/1_🏠_Home.py', encoding='utf-8').read())
elif st.session_state.page == 'detect':
    exec(open('pages/2_🎭_Detect_Emotion.py', encoding='utf-8').read())
elif st.session_state.page == 'music':
    exec(open('pages/3_🎵_Music_Therapy.py', encoding='utf-8').read())
elif st.session_state.page == 'therapy':
    exec(open('pages/4_🤖_AI_Therapy.py', encoding='utf-8').read())
elif st.session_state.page == 'about':
    exec(open('pages/5_ℹ️_About.py', encoding='utf-8').read())