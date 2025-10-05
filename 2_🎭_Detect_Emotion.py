# Detect Emotion Page
import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
from config import (
    GEMINI_API_KEY, GEMINI_MODEL, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    ELEVENLABS_API_URL, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, 
    EMOTION_TO_GENRE, EMOTION_TO_MOOD, BAD_MOODS
)
from hack_tracks import (
    ImprovedResNet, ResNetEmotionModel, SpotifyAPI, 
    GeminiMoodAnalyzer, ElevenLabsVoice, EMOTION_LABELS
)

def detect_emotion(image, model, device, face_cascade):
    """Detect emotion from image"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    
    if len(faces) == 0:
        return None, 0, None, image
    
    x, y, w, h = faces[0]
    face_gray = gray[y:y+h, x:x+w]
    
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    pil_img = Image.fromarray(face_gray)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    emotion = EMOTION_LABELS[predicted.item()]
    conf = confidence.item()
    probs = probabilities[0].cpu().numpy()
    
    color = (102, 126, 234)
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
    label = f"{emotion}: {conf*100:.0f}%"
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return emotion, conf, probs, image

# Header
st.markdown('<div class="section-header"><div class="section-title">🎭 Emotion Detection</div><div class="section-subtitle">Let AI understand how you\'re feeling</div></div>', unsafe_allow_html=True)

# Two column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="card-title">📸 Capture Your Moment</div>
        <div class="card-text">Your emotional state is the first step to wellness. Let our AI understand how you're feeling.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Camera input
    img_file = st.camera_input("📷 Take a live photo")
    
    # Or file upload
    if img_file is None:
        img_file = st.file_uploader("📁 Or upload an image", type=['jpg', 'jpeg', 'png'])
    
    if img_file is not None:
        image = Image.open(img_file)
        image_np = np.array(image)
        
        if st.button("🎯 DETECT EMOTION", use_container_width=True):
            with st.spinner("🧠 Analyzing your expression..."):
                emotion, confidence, probs, annotated_img = detect_emotion(
                    image_np.copy(),
                    st.session_state.emotion_model,
                    st.session_state.device,
                    st.session_state.face_cascade
                )
                
                if emotion:
                    st.session_state.current_emotion = emotion
                    st.session_state.confidence = confidence
                    st.session_state.probs = probs
                    st.session_state.annotated_img = annotated_img
                    st.success("✅ Emotion detected successfully!")
                else:
                    st.error("❌ No face detected. Please try again with a clearer photo.")

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="card-title">🎯 Your Emotional State</div>
        <div class="card-text">Our AI has analyzed your expression with precision and care.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if hasattr(st.session_state, 'current_emotion') and st.session_state.current_emotion:
        # Display emotion
        st.markdown(f"""
        <div class="emotion-display">
            <div class="emotion-label">{st.session_state.current_emotion.upper()}</div>
            <div class="emotion-confidence">{st.session_state.confidence*100:.1f}% Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show annotated image
        st.image(st.session_state.annotated_img, caption="✨ AI-Detected Expression", use_column_width=True)
        
        # Emotion probabilities
        with st.expander("📊 Detailed Emotion Analysis"):
            st.markdown('<div class="card-text">Our AI detected multiple emotional signals. Here\'s the complete breakdown:</div>', unsafe_allow_html=True)
            for idx, emotion_name in EMOTION_LABELS.items():
                prob = st.session_state.probs[idx] * 100
                st.progress(prob/100, text=f"{emotion_name.capitalize()}: {prob:.1f}%")
        
        # Next steps
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🎵 Get Music", use_container_width=True):
                st.session_state.page = 'music'
                st.rerun()
        with col_b:
            if st.button("🤖 Get Therapy", use_container_width=True):
                st.session_state.page = 'therapy'
                st.rerun()
    else:
        st.markdown("""
        <div class="alert alert-info">
            <strong>👆 Ready to begin?</strong><br/>
            Take or upload a photo to discover your emotional state and unlock your personalized wellness experience.
        </div>
        """, unsafe_allow_html=True)
