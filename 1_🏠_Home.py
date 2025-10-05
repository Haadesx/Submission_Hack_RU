# Home Page
import streamlit as st

st.markdown("""
<div class="hero-section">
    <div class="hero-title">🎵 HACK TRACKS</div>
    <div class="hero-subtitle">Transform Your Emotions into Wellness Through AI</div>
</div>
""", unsafe_allow_html=True)

# Features Grid
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">🎭</div>
        <div class="card-title">Emotion Detection</div>
        <div class="card-text">
            Advanced AI analyzes your facial expressions with 68% accuracy across 7 emotions
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">🎵</div>
        <div class="card-title">Music Therapy</div>
        <div class="card-text">
            Location-aware, personalized playlists powered by Gemini AI and Spotify
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">🤖</div>
        <div class="card-title">AI Therapy</div>
        <div class="card-text">
            Full conversational therapy sessions with voice narration for emotional support
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-header"><div class="section-title">How It Works</div><div class="section-subtitle">Simple, powerful, transformative</div></div>', unsafe_allow_html=True)

# Process Steps
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">1️⃣</div>
        <div class="card-title" style="font-size: 1.5rem;">Capture</div>
        <div class="card-text">
            Upload or take a photo of your face
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">2️⃣</div>
        <div class="card-title" style="font-size: 1.5rem;">Detect</div>
        <div class="card-text">
            AI analyzes your emotional state
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">3️⃣</div>
        <div class="card-title" style="font-size: 1.5rem;">Choose</div>
        <div class="card-text">
            Select music or therapy support
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">4️⃣</div>
        <div class="card-title" style="font-size: 1.5rem;">Transform</div>
        <div class="card-text">
            Experience emotional wellness
        </div>
    </div>
    """, unsafe_allow_html=True)

# Call to Action
st.markdown("<br><br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 GET STARTED", use_container_width=True):
        st.session_state.page = 'detect'
        st.rerun()

# Stats
st.markdown('<div class="section-header"><div class="section-title">Powered By</div></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🧠</div>
        <div class="card-title" style="font-size: 1.3rem;">Gemini AI</div>
        <div class="card-text" style="font-size: 0.95rem;">Song recommendations & therapy scripts</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎤</div>
        <div class="card-title" style="font-size: 1.3rem;">ElevenLabs</div>
        <div class="card-text" style="font-size: 0.95rem;">Natural voice narration</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎵</div>
        <div class="card-title" style="font-size: 1.3rem;">Spotify</div>
        <div class="card-text" style="font-size: 0.95rem;">Music streaming & discovery</div>
    </div>
    """, unsafe_allow_html=True)

