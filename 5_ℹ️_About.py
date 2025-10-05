# About Page
import streamlit as st

st.markdown('<div class="section-header"><div class="section-title">ℹ️ About Hack Tracks</div><div class="section-subtitle">Your personal AI wellness companion</div></div>', unsafe_allow_html=True)

# Mission
st.markdown("""
<div class="feature-card">
    <div class="card-title">🎯 Our Mission</div>
    <div class="card-text" style="font-size: 1.15rem;">
        Hack Tracks combines cutting-edge AI technology with music therapy and emotional support to create 
        a comprehensive wellness platform. We believe that everyone deserves access to emotional support, 
        and technology can be a bridge to better mental health.
    </div>
</div>
""", unsafe_allow_html=True)

# How it works
st.markdown('<div class="section-header"><div class="section-title">🔬 The Technology</div></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="card-title">🧠 Emotion Detection</div>
        <div class="card-text">
            <ul style="line-height: 2;">
                <li><strong>Model:</strong> ResNet-34 CNN</li>
                <li><strong>Accuracy:</strong> 68% on FER2013</li>
                <li><strong>Emotions:</strong> 7 categories</li>
                <li><strong>Training:</strong> 35,887 images</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <div class="card-title">🎤 Voice Synthesis</div>
        <div class="card-text">
            <ul style="line-height: 2;">
                <li><strong>Platform:</strong> ElevenLabs</li>
                <li><strong>Model:</strong> Flash v2.5</li>
                <li><strong>Quality:</strong> Natural voice</li>
                <li><strong>Features:</strong> Real-time TTS</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="card-title">🎵 Music Recommendations</div>
        <div class="card-text">
            <ul style="line-height: 2;">
                <li><strong>AI:</strong> Google Gemini 2.0</li>
                <li><strong>Source:</strong> Spotify API</li>
                <li><strong>Context:</strong> Location-aware</li>
                <li><strong>Variety:</strong> Always unique</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <div class="card-title">🤖 AI Therapy</div>
        <div class="card-text">
            <ul style="line-height: 2;">
                <li><strong>Engine:</strong> Gemini AI</li>
                <li><strong>Approach:</strong> Conversational</li>
                <li><strong>Techniques:</strong> CBT-based</li>
                <li><strong>Delivery:</strong> Voice + text</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Features breakdown
st.markdown('<div class="section-header"><div class="section-title">✨ Key Features</div></div>', unsafe_allow_html=True)

features = [
    ("🎭", "Real-time Emotion Detection", "Advanced AI analyzes facial expressions with high accuracy"),
    ("🌍", "Location-Aware", "Recommendations personalized for your city, culture, and time zone"),
    ("🔄", "Always Unique", "Every recommendation is different, ensuring fresh experiences"),
    ("💙", "Inclusive Support", "Therapy available for ALL emotions, not just negative ones"),
    ("🎧", "Voice Narration", "Professional text-to-speech for immersive experiences"),
    ("🎵", "Curated Playlists", "AI-selected music that matches your emotional state"),
]

col1, col2, col3 = st.columns(3)
columns = [col1, col2, col3]

for i, (icon, title, desc) in enumerate(features):
    with columns[i % 3]:
        st.markdown(f"""
        <div class="feature-card" style="text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">{icon}</div>
            <div class="card-title" style="font-size: 1.2rem;">{title}</div>
            <div class="card-text" style="font-size: 0.95rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Team & Credits
st.markdown('<div class="section-header"><div class="section-title">🏆 Built For</div></div>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-card" style="text-align: center;">
    <div style="font-size: 3rem; margin-bottom: 1rem;">🎓</div>
    <div class="card-title" style="font-size: 2rem;">RU Hacks 2025</div>
    <div class="card-text" style="font-size: 1.2rem;">
        <strong>Categories:</strong><br/>
        Social Good • Best Use of Gemini API • Best Use of ElevenLabs<br/>
        Best UI/UX Design • Education
    </div>
</div>
""", unsafe_allow_html=True)

# Tech Stack
st.markdown('<div class="section-header"><div class="section-title">🛠️ Tech Stack</div></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔥</div>
        <div class="card-title" style="font-size: 1.1rem;">PyTorch</div>
        <div class="card-text" style="font-size: 0.85rem;">Deep Learning</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎨</div>
        <div class="card-title" style="font-size: 1.1rem;">Streamlit</div>
        <div class="card-text" style="font-size: 0.85rem;">Web Framework</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🐍</div>
        <div class="card-title" style="font-size: 1.1rem;">Python</div>
        <div class="card-text" style="font-size: 0.85rem;">Core Language</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌐</div>
        <div class="card-title" style="font-size: 1.1rem;">APIs</div>
        <div class="card-text" style="font-size: 0.85rem;">Multiple Services</div>
    </div>
    """, unsafe_allow_html=True)

# Contact & Links
st.markdown('<div class="section-header"><div class="section-title">📞 Get Involved</div></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 2rem; margin-bottom: 1rem;">💻</div>
        <div class="card-title" style="font-size: 1.2rem;">GitHub</div>
        <div class="card-text">View source code<br/>and contribute</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 2rem; margin-bottom: 1rem;">📧</div>
        <div class="card-title" style="font-size: 1.2rem;">Feedback</div>
        <div class="card-text">Share your thoughts<br/>and suggestions</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <div style="font-size: 2rem; margin-bottom: 1rem;">🌟</div>
        <div class="card-title" style="font-size: 1.2rem;">Support</div>
        <div class="card-text">Help us grow<br/>and improve</div>
    </div>
    """, unsafe_allow_html=True)

