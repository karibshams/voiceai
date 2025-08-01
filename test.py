import streamlit as st
import os
import tempfile
from pathlib import Path
import json
from datetime import datetime
from app import VoiceMindAI, create_voicemind_instance
from prompt import get_prompt_by_style
import time
import pygame
import numpy as np

# Page configuration
st.set_page_config(
    page_title="VoiceMind™ Voice Chat Test",
    page_icon="🧠💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 15px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .ai-message {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 15px;
        border-left: 4px solid #9c27b0;
        margin: 1rem 0;
    }
    .crisis-alert {
        background-color: #ffebee;
        border: 2px solid #f44336;
        padding: 1rem;
        border-radius: 10px;
        color: #c62828;
    }
    .emotion-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables for voice chat"""
    if 'ai_instance' not in st.session_state:
        st.session_state.ai_instance = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {}
    if 'current_audio_path' not in st.session_state:
        st.session_state.current_audio_path = None
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False

def setup_user_profile():
    """Setup user profile in sidebar for personalized responses"""
    st.sidebar.header("👤 User Profile Setup")
    
    with st.sidebar.expander("📝 Personal Info", expanded=True):
        name = st.text_input("Name (optional)", value=st.session_state.user_profile.get("name", ""))
        age_group = st.selectbox(
            "Age Group", 
            ["", "teen", "young_adult", "adult", "senior"],
            index=["", "teen", "young_adult", "adult", "senior"].index(st.session_state.user_profile.get("age_group", ""))
        )
        gender = st.selectbox(
            "Gender (optional)", 
            ["", "male", "female", "non-binary", "prefer_not_to_say"],
            index=["", "male", "female", "non-binary", "prefer_not_to_say"].index(st.session_state.user_profile.get("gender", ""))
        )
    
    with st.sidebar.expander("🎯 Support Preferences"):
        support_style = st.radio(
            "Support Style",
            ["mental_health", "spiritual"],
            index=0 if st.session_state.user_profile.get("support_style", "mental_health") == "mental_health" else 1,
            format_func=lambda x: "Mental Health & Life Coaching" if x == "mental_health" else "Spiritual Growth & Faith-Based"
        )
        
        preferred_tone = st.selectbox(
            "Preferred Response Tone",
            ["", "gentle", "encouraging", "direct", "spiritual"],
            index=["", "gentle", "encouraging", "direct", "spiritual"].index(st.session_state.user_profile.get("preferred_tone", ""))
        )
        
        common_topics = st.multiselect(
            "Common Topics You Discuss",
            ["anxiety", "depression", "relationships", "work_stress", "family", "self_esteem", "grief", "bullying", "spiritual_growth"],
            default=st.session_state.user_profile.get("common_themes", [])
        )
    
    # Update profile
    st.session_state.user_profile = {
        "name": name,
        "age_group": age_group,
        "gender": gender,
        "support_style": support_style,
        "preferred_tone": preferred_tone,
        "common_themes": common_topics,
        "created_at": st.session_state.user_profile.get("created_at", datetime.now().isoformat())
    }
    
    # Remove empty values
    st.session_state.user_profile = {k: v for k, v in st.session_state.user_profile.items() if v}

def setup_ai_instance():
    """Setup AI instance with API key and user profile"""
    st.sidebar.header("🔑 API Configuration")
    
    api_key = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password",
        help="Enter your OpenAI API key to enable voice chat",
        value=os.getenv("OPENAI_API_KEY", "")
    )
    
    if api_key:
        try:
            if st.session_state.ai_instance is None or st.session_state.ai_instance.api_key != api_key:
                st.session_state.ai_instance = create_voicemind_instance(api_key, st.session_state.user_profile)
                st.sidebar.success("✅ VoiceMind AI Ready!")
            
            # Update user profile in AI instance
            st.session_state.ai_instance.set_user_profile(st.session_state.user_profile)
            return True
            
        except Exception as e:
            st.sidebar.error(f"❌ Failed to initialize AI: {str(e)}")
            return False
    else:
        st.sidebar.warning("⚠️ Please enter your OpenAI API key")
        return False

def display_main_header():
    """Display main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🧠💬 VoiceMind™ Voice Chat Test</h1>
        <p>Real-time voice-to-voice mental wellness coaching system</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "🟢 Ready" if st.session_state.ai_instance else "🔴 Setup Required"
        st.metric("AI Status", status)
    
    with col2:
        style = st.session_state.user_profile.get("support_style", "Not Set")
        display_style = "Mental Health" if style == "mental_health" else "Spiritual" if style == "spiritual" else "Not Set"
        st.metric("Support Style", display_style)
    
    with col3:
        chat_count = len(st.session_state.chat_history)
        st.metric("Conversations", chat_count)
    
    with col4:
        crisis_count = sum(1 for chat in st.session_state.chat_history if chat.get("crisis_detected", False))
        st.metric("Crisis Alerts", crisis_count)

def voice_chat_interface():
    """Main voice chat interface with LIVE MICROPHONE support"""
    st.header("🎙️ Live Voice Chat Session")
    
    if not st.session_state.ai_instance:
        st.error("❌ Please configure your OpenAI API key and user profile in the sidebar first.")
        return
    
    # Recording mode selection
    col1, col2 = st.columns(2)
    
    with col1:
        recording_mode = st.radio(
            "🎤 Recording Mode",
            ["live_auto", "push_to_talk", "file_upload"],
            format_func=lambda x: {
                "live_auto": "🔴 Live Auto (Voice Detection)",
                "push_to_talk": "🟡 Push-to-Talk (Fixed Duration)", 
                "file_upload": "📁 File Upload (Original)"
            }[x]
        )
    
    with col2:
        if recording_mode == "push_to_talk":
            ptt_duration = st.slider("Recording Duration (seconds)", 5, 30, 10)
        else:
            ptt_duration = 10
    
    # Live recording interface
    if recording_mode in ["live_auto", "push_to_talk"]:
        st.subheader("🎙️ Live Voice Recording")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔴 Start Recording", type="primary", use_container_width=True):
                # Call the live recording logic directly
                if recording_mode == "live_auto":
                    # Auto voice detection recording
                    support_style = st.session_state.user_profile.get("support_style", "mental_health")
                    result = st.session_state.ai_instance.start_live_conversation(
                        support_style=support_style,
                        callback=None  # You can implement a callback if needed
                    )
                else:
                    # Push-to-talk recording
                    support_style = st.session_state.user_profile.get("support_style", "mental_health")
                    st.info(f"🎤 Recording for {ptt_duration} seconds...")
                    audio_file = st.session_state.ai_instance.record_push_to_talk(ptt_duration)
                    st.info("🧠 Processing with AI...")
                    result = st.session_state.ai_instance.process_voice_conversation(audio_file, support_style)
                    if result["success"]:
                        st.info("🎵 Playing AI response...")
                        st.session_state.ai_instance.play_audio(result["response_audio_path"])
                        st.success("✅ Conversation complete!")
                    try:
                        os.remove(audio_file)
                    except:
                        pass
                # Add to chat history if successful
                if result["success"]:
                    chat_entry = {
                        "timestamp": datetime.now(),
                        "user_audio_transcript": result["transcript"],
                        "ai_response": result["ai_response"],
                        "emotion_analysis": result["emotion_analysis"],
                        "response_audio_path": result["response_audio_path"],
                        "mental_tools": result.get("mental_tools", []),
                        "crisis_detected": result.get("crisis_detected", False),
                        "bullying_detected": result.get("bullying_detected", False),
                        "conversation_id": result.get("conversation_id", ""),
                        "recording_mode": recording_mode
                    }
                    st.session_state.chat_history.append(chat_entry)
                    if result.get("crisis_detected"):
                        display_crisis_alert()
                    elif result.get("bullying_detected"):
                        display_bullying_alert()
        
        with col2:
            if st.button("⏹️ Stop Recording", use_container_width=True):
                if st.session_state.ai_instance:
                    st.session_state.ai_instance.stop_recording()
                    st.info("Recording stopped")
        
        with col3:
            if st.button("🎵 Test Microphone", use_container_width=True):
                test_microphone()
    
    else:
        # Original file upload interface
        st.subheader("📁 Upload Audio File")
        uploaded_audio = st.file_uploader(
            "Upload your voice journal audio file",
            type=['mp3', 'wav', 'm4a', 'ogg', 'flac'],
            help="Upload an audio file with your voice journal entry"
        )
        
        if uploaded_audio and st.button("🚀 Process Audio File", type="primary", use_container_width=True):
            process_voice_input(uploaded_audio, recording_mode="file_upload")
    
    # Display conversation history
    display_conversation_history()

def process_voice_input(uploaded_audio, recording_mode="file_upload", duration=10):
    """Process the uploaded voice input through VoiceMind AI (original file upload method or live modes)"""
    
    if recording_mode == "file_upload":
        with st.spinner("🎯 Processing your voice journal..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Save uploaded file
                status_text.text("📁 Saving audio file...")
                progress_bar.progress(10)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_audio.read())
                    temp_audio_path = tmp_file.name
                
                # Step 2: Process through VoiceMind AI
                status_text.text("🧠 Analyzing with VoiceMind AI...")
                progress_bar.progress(30)
                
                support_style = st.session_state.user_profile.get("support_style", "mental_health")
                result = st.session_state.ai_instance.process_voice_conversation(
                    audio_path=temp_audio_path,
                    support_style=support_style
                )
                
                progress_bar.progress(80)
                status_text.text("✅ Processing complete!")
                
                # Clean up temp file
                os.unlink(temp_audio_path)
                
                progress_bar.progress(100)
                time.sleep(0.5)  # Brief pause for UX
                progress_bar.empty()
                status_text.empty()
                
                if result["success"]:
                    # Add to chat history
                    chat_entry = {
                        "timestamp": datetime.now(),
                        "user_audio_transcript": result["transcript"],
                        "ai_response": result["ai_response"],
                        "emotion_analysis": result["emotion_analysis"],
                        "response_audio_path": result["response_audio_path"],
                        "mental_tools": result.get("mental_tools", []),
                        "crisis_detected": result.get("crisis_detected", False),
                        "bullying_detected": result.get("bullying_detected", False),
                        "recording_mode": "file_upload"
                    }
                    
                    st.session_state.chat_history.append(chat_entry)
                    
                    # Show success message
                    st.success("✅ Voice conversation processed successfully!")
                    
                    # Handle crisis situations
                    if result.get("crisis_detected"):
                        display_crisis_alert()
                    elif result.get("bullying_detected"):
                        display_bullying_alert()
                    
                    # Auto-play AI response
                    st.info("🎵 Playing AI response...")
                    
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error during voice processing: {str(e)}")
    else:
        # Start live recording based on selected mode
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        def recording_callback(status, data=None):
            """Callback for real-time recording updates"""
            if status == "recording_started":
                status_placeholder.success("🎤 Recording started! Speak now...")
            elif status == "recording":
                if data and "duration" in data:
                    progress_placeholder.progress(min(data["duration"] / 10, 1.0))
            elif status == "silence_detected":
                status_placeholder.info("🔇 Silence detected, processing...")
            elif status == "recording_complete":
                status_placeholder.success("✅ Recording complete!")
                progress_placeholder.empty()
            elif status == "recording_error":
                status_placeholder.error(f"❌ Recording failed: {data.get('error', 'Unknown error')}")
        
        def conversation_callback(status, data=None):
            """Callback for conversation processing updates"""
            if status == "conversation_started":
                status_placeholder.info("🚀 Starting live conversation...")
            elif status == "processing_started":
                status_placeholder.info("🧠 Processing with AI...")
            elif status == "processing_complete":
                status_placeholder.success("✅ AI processing complete!")
            elif status == "playing_response":
                status_placeholder.info("🎵 Playing AI response...")
            elif status == "response_played":
                status_placeholder.success("✅ Conversation complete!")
            elif status == "conversation_error":
                status_placeholder.error(f"❌ Conversation failed: {data.get('error', 'Unknown error')}")
        
        try:
            support_style = st.session_state.user_profile.get("support_style", "mental_health")
            
            if recording_mode == "live_auto":
                # Auto voice detection recording
                result = st.session_state.ai_instance.start_live_conversation(
                    support_style=support_style,
                    callback=conversation_callback
                )
            else:
                # Push-to-talk recording
                status_placeholder.info(f"🎤 Recording for {duration} seconds...")
                audio_file = st.session_state.ai_instance.record_push_to_talk(duration)
                
                status_placeholder.info("🧠 Processing with AI...")
                result = st.session_state.ai_instance.process_voice_conversation(audio_file, support_style)
                
                if result["success"]:
                    status_placeholder.info("🎵 Playing AI response...")
                    st.session_state.ai_instance.play_audio(result["response_audio_path"])
                    status_placeholder.success("✅ Conversation complete!")
                
                # Clean up
                try:
                    os.remove(audio_file)
                except:
                    pass
            
            # Add to chat history if successful
            if result["success"]:
                chat_entry = {
                    "timestamp": datetime.now(),
                    "user_audio_transcript": result["transcript"],
                    "ai_response": result["ai_response"],
                    "emotion_analysis": result["emotion_analysis"],
                    "response_audio_path": result["response_audio_path"],
                    "mental_tools": result.get("mental_tools", []),
                    "crisis_detected": result.get("crisis_detected", False),
                    "bullying_detected": result.get("bullying_detected", False),
                    "conversation_id": result.get("conversation_id", ""),
                    "recording_mode": recording_mode
                }
                
                st.session_state.chat_history.append(chat_entry)
                
                # Handle crisis situations
                if result.get("crisis_detected"):
                    display_crisis_alert()
                elif result.get("bullying_detected"):
                    display_bullying_alert()
                
                # Clear status after delay
                time.sleep(2)
                status_placeholder.empty()
                progress_placeholder.empty()
            
        except Exception as e:
            status_placeholder.error(f"❌ Live recording failed: {str(e)}")

def test_microphone():
    """Test microphone functionality"""
    st.info("🎤 Testing microphone for 3 seconds...")
    
    try:
        import pyaudio
        
        audio = pyaudio.PyAudio()
        
        # Test if microphone is accessible
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        # Record for 3 seconds
        for i in range(int(16000 / 1024 * 3)):
            data = stream.read(1024)
            # Check if we're getting audio data
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            
            if i % 10 == 0:  # Update every ~100ms
                st.write(f"Volume level: {volume}")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        st.success("✅ Microphone test successful!")
        
    except Exception as e:
        st.error(f"❌ Microphone test failed: {str(e)}")
        st.write("**Troubleshooting:**")
        st.write("1. Check microphone permissions")
        st.write("2. Install: `pip install pyaudio`")
        st.write("3. On macOS: `brew install portaudio`")
        st.write("4. On Ubuntu: `sudo apt install python3-pyaudio`")

def display_conversation_history():
    """Display the conversation history in chat format"""
    if not st.session_state.chat_history:
        st.info("💬 No conversations yet. Upload a voice journal to start chatting with VoiceMind!")
        return
    
    st.subheader("💬 Conversation History")
    
    # Show recent conversations (last 5)
    recent_chats = st.session_state.chat_history[-5:]
    
    for i, chat in enumerate(reversed(recent_chats)):
        with st.container():
            # Timestamp
            timestamp = chat["timestamp"].strftime("%I:%M %p - %b %d")
            st.caption(f"🕐 {timestamp}")
            
            # User message
            st.markdown(f"""
            <div class="user-message">
                <strong>🗣️ You said:</strong><br>
                "{chat['user_audio_transcript']}"
            </div>
            """, unsafe_allow_html=True)
            
            # Emotion analysis
            emotion_data = chat["emotion_analysis"]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                emotion = emotion_data.get("primary_emotion", "neutral")
                st.markdown(f"""
                <div class="emotion-card">
                    <strong>🎭 Emotion:</strong> {emotion.title()}<br>
                    <strong>📊 Intensity:</strong> {emotion_data.get("intensity", "medium").title()}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                tone = emotion_data.get("tone", "neutral")
                needs_support = "Yes" if emotion_data.get("needs_support", False) else "No"
                st.markdown(f"""
                <div class="emotion-card">
                    <strong>🎵 Tone:</strong> {tone.title()}<br>
                    <strong>🤗 Needs Support:</strong> {needs_support}
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                crisis = "🚨 Yes" if chat.get("crisis_detected", False) else "✅ No"
                bullying = "🛡️ Yes" if chat.get("bullying_detected", False) else "✅ No"
                st.markdown(f"""
                <div class="emotion-card">
                    <strong>Crisis:</strong> {crisis}<br>
                    <strong>Bullying:</strong> {bullying}
                </div>
                """, unsafe_allow_html=True)
            
            # AI response
            st.markdown(f"""
            <div class="ai-message">
                <strong>🧠 VoiceMind responded:</strong><br>
                {chat['ai_response']}
            </div>
            """, unsafe_allow_html=True)
            
            # Audio controls and mental health tools
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if st.button(f"🎵 Play AI Response", key=f"play_{i}"):
                    try:
                        if os.path.exists(chat["response_audio_path"]):
                            pygame.mixer.music.load(chat["response_audio_path"])
                            pygame.mixer.music.play()
                            st.success("🎵 Playing audio...")
                        else:
                            st.warning("Audio file not found")
                    except Exception as e:
                        st.error(f"Playback failed: {str(e)}")
            
            with col2:
                # Mental health tools
                if chat.get("mental_tools"):
                    with st.expander("🧘 Suggested Mental Health Tools"):
                        for tool in chat["mental_tools"]:
                            st.write(f"**{tool['name']}**")
                            st.write(tool['instruction'])
                            st.write("---")
            
            st.markdown("---")

def display_crisis_alert():
    """Display crisis intervention alert"""
    st.markdown("""
    <div class="crisis-alert">
        <h3>🚨 CRISIS SUPPORT ACTIVATED</h3>
        <p><strong>VoiceMind detected concerning content in your message.</strong></p>
        
        <p><strong>Immediate Resources:</strong></p>
        <ul>
            <li>🆘 <strong>National Suicide Prevention Lifeline:</strong> 988</li>
            <li>💬 <strong>Crisis Text Line:</strong> Text HOME to 741741</li>
            <li>🚨 <strong>Emergency Services:</strong> 911</li>
        </ul>
        
        <p><strong>Remember:</strong> You matter. Your life has value. Help is available 24/7.</p>
    </div>
    """, unsafe_allow_html=True)

def display_bullying_alert():
    """Display anti-bullying support alert"""
    st.markdown("""
    <div style="background-color: #fff3e0; border: 2px solid #ff9800; padding: 1rem; border-radius: 10px; color: #e65100;">
        <h3>🛡️ ANTI-BULLYING SUPPORT ACTIVATED</h3>
        <p><strong>VoiceMind detected that you may be experiencing bullying.</strong></p>
        
        <p><strong>Important Reminders:</strong></p>
        <ul>
            <li>✊ This is NOT your fault</li>
            <li>💪 You are brave for speaking up</li>
            <li>🤝 You deserve respect and kindness</li>
            <li>📞 It's okay to ask for help from trusted adults</li>
        </ul>
        
        <p><strong>Resources:</strong></p>
        <ul>
            <li>📞 <strong>StopBullying.gov</strong> for guidance and resources</li>
            <li>💬 <strong>Crisis Text Line:</strong> Text HOME to 741741</li>
            <li>🏫 Tell a trusted teacher, counselor, or parent</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def conversation_analytics():
    """Display conversation analytics and insights"""
    st.header("📊 Conversation Analytics")
    
    if not st.session_state.chat_history:
        st.info("No conversation data available yet.")
        return
    
    if st.session_state.ai_instance:
        summary = st.session_state.ai_instance.get_conversation_summary()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Sessions", summary["total_sessions"])
            st.metric("Crisis Incidents", summary["crisis_incidents"])
        
        with col2:
            if summary["dominant_emotions"]:
                st.write("**Top Emotions:**")
                for emotion, count in summary["dominant_emotions"]:
                    st.write(f"• {emotion.title()}: {count} times")
        
        with col3:
            if summary["recent_mood_trend"]:
                st.write("**Recent Mood Trend:**")
                mood_trend = " → ".join([m.title() for m in summary["recent_mood_trend"]])
                st.write(mood_trend)
        
        # Mood trend chart (simple)
        if len(st.session_state.chat_history) > 1:
            st.subheader("📈 Emotional Journey")
            
            emotions = [chat["emotion_analysis"].get("primary_emotion", "neutral") for chat in st.session_state.chat_history]
            timestamps = [chat["timestamp"].strftime("%m/%d %H:%M") for chat in st.session_state.chat_history]
            
            # Create simple emotion trend display
            emotion_data = []
            for i, (emotion, timestamp) in enumerate(zip(emotions, timestamps)):
                emotion_data.append({"Session": i+1, "Emotion": emotion.title(), "Time": timestamp})
            
            st.write("**Emotion Timeline:**")
            for entry in emotion_data[-10:]:  # Last 10 sessions
                st.write(f"Session {entry['Session']} ({entry['Time']}): {entry['Emotion']}")

def sidebar_info():
    """Display helpful information in sidebar"""
    st.sidebar.header("ℹ️ About VoiceMind™")
    
    st.sidebar.info("""
    **VoiceMind™ Features:**
    
    🎙️ **Voice-to-Text:** OpenAI Whisper transcription
    
    🧠 **AI Coaching:** GPT-4 powered responses
    
    🎭 **Emotion Analysis:** Real-time mood detection
    
    🚨 **Crisis Detection:** Automatic safety alerts
    
    🛡️ **Anti-Bullying:** Specialized support
    
    🧘 **Mental Tools:** Personalized coping strategies
    
    🔊 **Text-to-Speech:** Spoken AI responses
    
    💬 **Context Memory:** Conversation awareness
    """)
    
    st.sidebar.header("🔧 Test Guidelines")
    st.sidebar.write("""
    **For Testing:**
    1. Set up your profile first
    2. Upload clear audio files
    3. Speak naturally about your feelings
    4. Test different emotional states
    5. Try both support styles
    6. Test crisis keywords (safely)
    """)
    
    if st.sidebar.button("🗑️ Clear All Data"):
        st.session_state.chat_history = []
        st.session_state.user_profile = {}
        if st.session_state.ai_instance:
            st.session_state.ai_instance.conversation_history = []
        st.sidebar.success("Data cleared!")

def main():
    """Main application function"""
    initialize_session_state()
    display_main_header()
    
    # Setup user profile and AI
    setup_user_profile()
    ai_ready = setup_ai_instance()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Voice Chat", "📊 Analytics"])
    
    with tab1:
        voice_chat_interface()
    
    with tab2:
        conversation_analytics()
    
    # Sidebar information
    sidebar_info()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 1rem;'>
            <p>🧠💬 <strong>VoiceMind™ Voice Chat Test Interface</strong></p>
            <p>Real-time voice-to-voice mental wellness system with crisis detection & personalized coaching</p>
            <p><em>Built for developers to test complete AI pipeline functionality</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    