# Music Therapy Page
import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
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

def get_spotify_recommendations(emotion, gemini, spotify, location=None, seed=None):
    """Get music recommendations"""
    location_context = ""
    if location:
        time_of_day = datetime.now().strftime("%I:%M %p")
        location_context = f" The person is in {location['city']}, {location['country']} at {time_of_day}."
    
    variation_seed = f"{seed}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    prompt = f"""As a music therapy expert, suggest 15 specific real songs (with artist names) 
that would be therapeutic for someone feeling {emotion}.{location_context}

Consider:
- Local music culture and preferences in {location['country'] if location else 'their region'}
- Time of day: {datetime.now().strftime('%I %p')}
- Regional popular artists that match the mood
- Variety: Make these recommendations UNIQUE (seed: {variation_seed[:20]})

Format each as: "Song Title - Artist Name"
Make them diverse and culturally relevant!"""

    try:
        response = gemini.model.generate_content(prompt)
        lines = response.text.strip().split('\n')
        gemini_songs = []
        for line in lines:
            line = line.lstrip('0123456789.- ')
            if ' - ' in line and len(line) > 5:
                gemini_songs.append(line)
        gemini_songs = gemini_songs[:15]
    except:
        gemini_songs = []
    
    if gemini_songs:
        tracks = []
        for song in gemini_songs[:12]:
            search_results = spotify.search_tracks(song, limit=1)
            if search_results:
                tracks.extend(search_results)
        
        if tracks:
            random.shuffle(tracks)
            return tracks[:10]
    
    moods = EMOTION_TO_MOOD.get(emotion, ['music'])
    random.shuffle(moods)
    search_query = f"{emotion} {moods[0]} {location['country'] if location else ''} therapy"
    tracks = spotify.search_tracks(search_query, limit=20)
    random.shuffle(tracks)
    return tracks[:10]

# Header
st.markdown('<div class="section-header"><div class="section-title">🎵 Music Therapy</div><div class="section-subtitle">Personalized playlists for your emotional journey</div></div>', unsafe_allow_html=True)

if not st.session_state.current_emotion:
    st.markdown("""
    <div class="alert alert-warning">
        <strong>⚠️ Emotion Not Detected</strong><br/>
        Please go to the Detect Emotion page first to analyze your mood.
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎭 Go to Emotion Detection", use_container_width=True):
        st.session_state.page = 'detect'
        st.rerun()
else:
    # Emotion-specific message
    emotion_messages = {
        'happy': '✨ You\'re radiating positive energy! Let\'s amplify that joy with the perfect soundtrack.',
        'sad': '💙 Every emotion has value. Let\'s find comfort and healing through music.',
        'angry': '🔥 Your feelings are valid. Let\'s channel that energy into powerful music.',
        'fearful': '🌟 You\'re brave for acknowledging your feelings. Let\'s find calm together.',
        'surprised': '🎭 Embrace the unexpected! Let\'s celebrate this moment with uplifting music.',
        'disgusted': '💚 It\'s okay to feel this way. Let\'s find clarity through music.',
        'neutral': '🎯 Perfect balance. Let\'s enhance your day with personalized recommendations.'
    }
    
    message = emotion_messages.get(st.session_state.current_emotion, 'Let me help you find the perfect music.')
    st.markdown(f'<div class="alert alert-info" style="text-align: center; font-size: 1.2rem;">{message}</div>', unsafe_allow_html=True)
    
    # Current mood display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="emotion-display" style="padding: 2rem;">
            <div class="emotion-label" style="font-size: 3rem;">{st.session_state.current_emotion.upper()}</div>
            <div class="emotion-confidence" style="font-size: 1.5rem;">{st.session_state.confidence*100:.1f}% Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Get recommendations button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎵 GET PERSONALIZED PLAYLIST", use_container_width=True):
            st.session_state.recommendation_seed = random.randint(0, 1000000)
            
            with st.spinner("🎼 Asking Gemini AI for personalized song suggestions..."):
                tracks = get_spotify_recommendations(
                    st.session_state.current_emotion,
                    st.session_state.gemini,
                    st.session_state.spotify,
                    location=st.session_state.location,
                    seed=st.session_state.recommendation_seed
                )
                st.session_state.current_tracks = tracks
                
                if st.session_state.location:
                    st.success(f"✅ Personalized for {st.session_state.location['city']}!")
    
    # Display tracks
    if st.session_state.current_tracks:
        st.markdown('<div class="section-header"><div class="section-title">🎼 Your Curated Playlist</div><div class="section-subtitle">Specially selected for your emotional journey</div></div>', unsafe_allow_html=True)
        
        for i, track in enumerate(st.session_state.current_tracks[:10], 1):
            st.markdown(f"""
            <div class="track-card">
                <div style="font-size: 1.3rem; font-weight: 700; color: #2d3748; margin-bottom: 0.5rem;">
                    {i}. {track['name']}
                </div>
                <div style="color: #667eea; font-weight: 600; margin-bottom: 0.3rem;">
                    by {track['artist']}
                </div>
                <div style="color: #6b7280; font-size: 0.95rem;">
                    Album: {track['album']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if track['url']:
                st.link_button(f"▶️ Play on Spotify", track['url'], use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)

