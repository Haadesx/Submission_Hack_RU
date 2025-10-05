# AI Therapy Page
import streamlit as st
import os
import time
from datetime import datetime
from config import (
    GEMINI_API_KEY, GEMINI_MODEL, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    ELEVENLABS_API_URL
)
from hack_tracks import GeminiMoodAnalyzer, ElevenLabsVoice

def create_direct_therapy_script(emotion, gemini, location=None):
    """Create a therapy script that starts directly with therapeutic content"""
    location_context = ""
    if location:
        location_context = f" in {location['city']}, {location['country']}"
    
    # Create a direct therapy script using Gemini - NO INTRO TEXT
    prompt = f"""Create a warm, direct therapy session script for someone feeling {emotion}{location_context}.

IMPORTANT: Start directly with the therapeutic content. Do NOT include any introductory text like "Here's the script:" or "Here's the audio message script:" or "Okay, here's".

Begin immediately with the therapy session content as if you are speaking directly to the person.

Include:
1. Direct warm greeting and acknowledgment of their emotion
2. Breathing exercise guidance  
3. Positive affirmations
4. Music therapy benefits
5. Encouraging closing

Write it as a script to be read aloud, natural and conversational.
Consider their location/culture: {location['country'] if location else 'their region'}
Time: {datetime.now().strftime('%A, %I:%M %p')}

Start directly with the therapeutic content - no labels or introductions. Just begin speaking to them."""

    try:
        response = gemini.model.generate_content(prompt)
        script = response.text.strip()
        
        # Clean up any remaining intro text that might slip through
        lines = script.split('\n')
        cleaned_lines = []
        skip_intro = True
        
        for line in lines:
            stripped = line.strip()
            # Skip lines that are clearly intro text
            if skip_intro and (
                stripped.startswith(("Here's", "Here is", "Okay, here's", "Here's the", "This is", "**", "Okay,"))
                or "script:" in stripped.lower()
                or "audio message" in stripped.lower()
            ):
                continue
            else:
                skip_intro = False
                cleaned_lines.append(line)
        
        script = '\n'.join(cleaned_lines).strip()
        return script
    except Exception as e:
        return f"Error creating therapy script: {e}"

# Main content
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color: #2E86AB; margin-bottom: 0.5rem;">🤖 AI Therapy Session</h1>
    <p style="color: #666; font-size: 1.1rem;">Personalized therapeutic support for your emotional wellness</p>
</div>
""", unsafe_allow_html=True)

# Check if we have the required components
if 'gemini' not in st.session_state or 'voice' not in st.session_state:
    st.error("⚠️ AI components not initialized. Please go to the Home page first.")
    st.stop()

# Get current emotion
current_emotion = st.session_state.get('current_emotion', None)

# Check if emotion has been detected
if current_emotion is None:
    st.warning("⚠️ No emotion detected yet. Please detect your emotion first!")
    st.info("👉 Go to **Detect Emotion** page to analyze your facial expression, then come back here for personalized therapy.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎭 Go to Detect Emotion", use_container_width=True, type="primary"):
            st.session_state.page = 'detect'
            st.rerun()
    
    st.stop()

# Display current emotion
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;">
        <h3 style="color: white; margin: 0;">Detected Emotion: {current_emotion.title()}</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Therapy session personalized for your {current_emotion} emotion
        </p>
    </div>
    """, unsafe_allow_html=True)

# Therapy options
st.markdown("### 🎯 Choose Your Therapy Session")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745;">
        <h4 style="color: #28a745; margin-top: 0;">🎧 Quick Therapy Session</h4>
        <p style="margin-bottom: 0;">A brief 1-2 minute therapeutic message with breathing exercises and affirmations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎧 Generate Quick Session", use_container_width=True, key="quick_session"):
        with st.spinner("Creating your personalized therapy session..."):
            # Generate direct therapy script
            script = create_direct_therapy_script(
                current_emotion, 
                st.session_state.gemini, 
                st.session_state.get('location')
            )
            
            if script and not script.startswith("Error"):
                # Generate voice message
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"voice_messages/therapy_{current_emotion}_{timestamp}.mp3"
                os.makedirs("voice_messages", exist_ok=True)
                
                st.session_state.voice.speak(script, filename)
                
                st.success("✅ Therapy session generated!")
                
                # Display the script
                st.markdown("### 📜 Your Therapy Script")
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                            border-left: 4px solid #007bff; margin: 1rem 0;">
                    <div style="white-space: pre-wrap; font-family: 'Inter', sans-serif; line-height: 1.6;">
{script}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Audio player
                if os.path.exists(filename):
                    st.audio(filename, format="audio/mp3")
                    st.info(f"🎧 Audio saved as: {filename}")
            else:
                st.error(f"❌ Failed to generate therapy session: {script}")

with col2:
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #dc3545;">
        <h4 style="color: #dc3545; margin-top: 0;">🧠 Deep Therapy Session</h4>
        <p style="margin-bottom: 0;">A comprehensive 3-5 minute session with detailed therapeutic techniques and personalized guidance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🧠 Generate Deep Session", use_container_width=True, key="deep_session"):
        with st.spinner("Creating your comprehensive therapy session..."):
            # Generate longer, more detailed therapy script
            location_context = ""
            if st.session_state.get('location'):
                location_context = f" in {st.session_state.location['city']}, {st.session_state.location['country']}"
            
            prompt = f"""Create a comprehensive 3-5 minute therapy session script for someone feeling {current_emotion}{location_context}.

IMPORTANT: Start directly with the therapeutic content. Do NOT include any introductory text like "Here's the script:" or "Here's the audio message script:" or "Okay, here's".

Begin immediately with the therapy session content as if you are speaking directly to the person.

Include:
1. Direct warm greeting and acknowledgment of their emotion
2. Detailed breathing exercise guidance (30-60 seconds)
3. Multiple positive affirmations specific to their emotion
4. Cognitive reframing techniques
5. Music therapy benefits and recommendations
6. Practical coping strategies
7. Encouraging closing with next steps

Write it as a script to be read aloud, natural and conversational.
Consider their location/culture: {st.session_state.get('location', {}).get('country', 'their region')}
Time: {datetime.now().strftime('%A, %I:%M %p')}

Start directly with the therapeutic content - no labels or introductions. Just begin speaking to them."""

            try:
                response = st.session_state.gemini.model.generate_content(prompt)
                script = response.text.strip()
                
                # Clean up any remaining intro text
                lines = script.split('\n')
                cleaned_lines = []
                skip_intro = True
                
                for line in lines:
                    stripped = line.strip()
                    # Skip lines that are clearly intro text
                    if skip_intro and (
                        stripped.startswith(("Here's", "Here is", "Okay, here's", "Here's the", "This is", "**", "Okay,"))
                        or "script:" in stripped.lower()
                        or "audio message" in stripped.lower()
                    ):
                        continue
                    else:
                        skip_intro = False
                        cleaned_lines.append(line)
                
                script = '\n'.join(cleaned_lines).strip()
                
                if script:
                    # Generate voice message
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"voice_messages/deep_therapy_{current_emotion}_{timestamp}.mp3"
                    os.makedirs("voice_messages", exist_ok=True)
                    
                    st.session_state.voice.speak(script, filename)
                    
                    st.success("✅ Deep therapy session generated!")
                    
                    # Display the script
                    st.markdown("### 📜 Your Deep Therapy Script")
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                                border-left: 4px solid #dc3545; margin: 1rem 0;">
                        <div style="white-space: pre-wrap; font-family: 'Inter', sans-serif; line-height: 1.6;">
{script}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Audio player
                    if os.path.exists(filename):
                        st.audio(filename, format="audio/mp3")
                        st.info(f"🎧 Audio saved as: {filename}")
                else:
                    st.error("❌ Failed to generate deep therapy session")
                    
            except Exception as e:
                st.error(f"❌ Error creating deep therapy session: {e}")

# Additional features
st.markdown("---")
st.markdown("### 🌟 Additional Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
        <h4 style="color: #007bff;">Personalized</h4>
        <p style="font-size: 0.9rem; color: #666;">Tailored to your current emotion and location</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔬</div>
        <h4 style="color: #28a745;">Evidence-Based</h4>
        <p style="font-size: 0.9rem; color: #666;">Uses proven therapeutic techniques</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌍</div>
        <h4 style="color: #dc3545;">Culturally Aware</h4>
        <p style="font-size: 0.9rem; color: #666;">Considers your location and culture</p>
    </div>
    """, unsafe_allow_html=True)
