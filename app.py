import openai
import os
from gtts import gTTS
import tempfile
import pygame
from typing import Optional, Dict, Any, List
import json
import re
from datetime import datetime
import uuid
import pyaudio
import wave
import threading
import time
from queue import Queue
import numpy as np

class VoiceMindAI:
    """
    VoiceMind AI - Complete voice-to-voice mental wellness system
    Features: STT, emotion analysis, crisis detection, personalized coaching, TTS, LIVE MICROPHONE
    """
    
    def __init__(self, api_key: str):
        """Initialize VoiceMind AI with OpenAI API key"""
        self.api_key = api_key
        openai.api_key = api_key
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Live recording configuration
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.recording = False
        self.audio_data = []
        self.silence_threshold = 500  # Adjust based on environment
        self.silence_duration = 2.0  # Seconds of silence to stop recording
        
        # User context and conversation memory
        self.user_profile = {}
        self.conversation_history = []
        self.crisis_keywords = [
            "suicide", "kill myself", "end it all", "hurt myself", "cut myself",
            "worthless", "nobody cares", "give up", "can't go on", "hopeless",
            "bullying", "bullied", "harassment", "threats", "scared", "unsafe"
        ]
        
        # Mental health tools based on emotions
        self.mental_tools = {
            "anxiety": ["breathing_exercise", "grounding_5_4_3_2_1", "progressive_muscle_relaxation"],
            "sadness": ["gratitude_practice", "self_compassion", "gentle_movement"],
            "anger": ["breathing_exercise", "progressive_muscle_relaxation", "mindful_observation"],
            "stress": ["breathing_exercise", "grounding_5_4_3_2_1", "mindful_break"],
            "fear": ["grounding_5_4_3_2_1", "safety_affirmations", "breathing_exercise"],
            "overwhelmed": ["simplify_tasks", "breathing_exercise", "priority_setting"],
            "lonely": ["connection_reminders", "self_compassion", "gratitude_practice"]
        }
    
    def set_user_profile(self, profile: Dict[str, Any]) -> None:
        """Set user profile for personalized responses"""
        self.user_profile = profile
        
    def add_to_conversation_history(self, user_input: str, ai_response: str, emotion_data: Dict) -> None:
        """Add conversation to history for context awareness"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response,
            "emotion_data": emotion_data,
            "session_id": str(uuid.uuid4())[:8]
        }
        self.conversation_history.append(entry)
        
        # Keep only last 10 conversations for context
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def is_silent(self, audio_chunk) -> bool:
        """Check if audio chunk is silent (below threshold)"""
        try:
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            return np.abs(audio_data).mean() < self.silence_threshold
        except:
            return True
    
    def record_live_audio(self, max_duration: int = 30, callback=None) -> str:
        """
        Record live audio from microphone with automatic silence detection
        
        Args:
            max_duration (int): Maximum recording duration in seconds
            callback (function): Optional callback for real-time updates
            
        Returns:
            str: Path to recorded audio file
        """
        audio = pyaudio.PyAudio()
        self.recording = True
        self.audio_data = []
        
        try:
            # Open audio stream
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            print("🎤 Recording started... Speak now!")
            if callback:
                callback("recording_started")
            
            silence_frames = 0
            max_frames = int(self.RATE / self.CHUNK * max_duration)
            silence_threshold_frames = int(self.RATE / self.CHUNK * self.silence_duration)
            
            for i in range(max_frames):
                if not self.recording:
                    break
                    
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                self.audio_data.append(data)
                
                # Check for silence
                if self.is_silent(data):
                    silence_frames += 1
                else:
                    silence_frames = 0
                
                # Stop if silence detected for too long (after initial speech)
                if len(self.audio_data) > silence_threshold_frames and silence_frames >= silence_threshold_frames:
                    print("🔇 Silence detected, stopping recording...")
                    if callback:
                        callback("silence_detected")
                    break
                
                # Real-time callback
                if callback and i % 10 == 0:  # Update every ~100ms
                    callback("recording", {"duration": i * self.CHUNK / self.RATE})
            
            stream.stop_stream()
            stream.close()
            
            # Save recorded audio to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"live_recording_{timestamp}.wav"
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(self.audio_data))
            
            print(f"🎵 Recording saved: {filename}")
            if callback:
                callback("recording_complete", {"filename": filename})
            
            return filename
            
        except Exception as e:
            print(f"❌ Recording failed: {str(e)}")
            if callback:
                callback("recording_error", {"error": str(e)})
            raise e
        finally:
            audio.terminate()
            self.recording = False
    
    def stop_recording(self):
        """Stop the current recording"""
        self.recording = False
        print("⏹️ Recording stopped by user")
    
    def record_push_to_talk(self, duration: int = 10) -> str:
        """
        Record audio for a fixed duration (push-to-talk style)
        
        Args:
            duration (int): Recording duration in seconds
            
        Returns:
            str: Path to recorded audio file
        """
        audio = pyaudio.PyAudio()
        self.audio_data = []
        
        try:
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            print(f"🎤 Recording for {duration} seconds... Speak now!")
            
            frames = int(self.RATE / self.CHUNK * duration)
            for i in range(frames):
                data = stream.read(self.CHUNK)
                self.audio_data.append(data)
                
                # Progress indicator
                if i % (frames // 10) == 0:
                    progress = (i / frames) * 100
                    print(f"📊 Recording: {progress:.0f}%")
            
            stream.stop_stream()
            stream.close()
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ptt_recording_{timestamp}.wav"
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(self.audio_data))
            
            print(f"✅ Push-to-talk recording complete: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Push-to-talk recording failed: {str(e)}")
            raise e
        finally:
            audio.terminate()
    
    def start_live_conversation(self, support_style: str = "mental_health", callback=None) -> Dict[str, Any]:
        """
        Complete live voice conversation - record → process → respond
        
        Args:
            support_style (str): Type of support ("mental_health" or "spiritual")
            callback (function): Optional callback for real-time updates
            
        Returns:
            Dict: Complete conversation results
        """
        try:
            if callback:
                callback("conversation_started")
            
            # Step 1: Record live audio
            def recording_callback(status, data=None):
                if callback:
                    callback(f"recording_{status}", data)
            
            audio_file = self.record_live_audio(callback=recording_callback)
            
            if callback:
                callback("processing_started")
            
            # Step 2: Process through existing pipeline
            result = self.process_voice_conversation(audio_file, support_style)
            
            if callback:
                callback("processing_complete", result)
            
            # Step 3: Auto-play response
            if result["success"] and os.path.exists(result["response_audio_path"]):
                if callback:
                    callback("playing_response")
                self.play_audio(result["response_audio_path"])
                if callback:
                    callback("response_played")
            
            # Clean up temp audio file
            try:
                os.remove(audio_file)
            except:
                pass
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Live conversation failed: {str(e)}",
                "crisis_detected": False,
                "bullying_detected": False
            }
            if callback:
                callback("conversation_error", error_result)
            return error_result
        """Transcribe audio file to text using OpenAI Whisper"""
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript.strip()
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")
    
    def analyze_emotion_and_crisis(self, text: str) -> Dict[str, Any]:
        """Analyze emotional content and detect crisis situations"""
        
        # Crisis detection first
        crisis_detected = any(keyword in text.lower() for keyword in self.crisis_keywords)
        crisis_level = "high" if crisis_detected else "none"
        
        # Bullying detection
        bullying_keywords = ["bullying", "bullied", "harassment", "threats", "making fun", "picked on"]
        bullying_detected = any(keyword in text.lower() for keyword in bullying_keywords)
        
        emotion_prompt = f"""
        Analyze this text for emotional content and return JSON:
        
        Text: "{text}"
        
        Return ONLY valid JSON:
        {{
            "primary_emotion": "dominant emotion (anxiety/sadness/anger/joy/fear/stress/overwhelmed/lonely/neutral)",
            "secondary_emotions": ["emotion1", "emotion2"],
            "intensity": "low/medium/high",
            "tone": "positive/neutral/negative",
            "mood_indicators": ["indicator1", "indicator2"],
            "emotional_summary": "brief summary of emotional state",
            "needs_support": true/false,
            "support_type": "crisis/emotional/practical/spiritual"
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an emotion analysis AI. Return only valid JSON."},
                    {"role": "user", "content": emotion_prompt}
                ],
                temperature=0.3,
                max_tokens=250
            )
            
            emotion_data = json.loads(response.choices[0].message.content)
            
            # Add crisis and bullying detection results
            emotion_data["crisis_detected"] = crisis_detected
            emotion_data["crisis_level"] = crisis_level
            emotion_data["bullying_detected"] = bullying_detected
            
            return emotion_data
            
        except Exception as e:
            return {
                "primary_emotion": "neutral",
                "secondary_emotions": [],
                "intensity": "medium",
                "tone": "neutral",
                "mood_indicators": [],
                "emotional_summary": "Unable to analyze emotion",
                "needs_support": False,
                "support_type": "emotional",
                "crisis_detected": crisis_detected,
                "crisis_level": crisis_level,
                "bullying_detected": bullying_detected
            }
    
    def get_mental_health_tools(self, primary_emotion: str) -> List[Dict[str, str]]:
        """Get personalized mental health tools based on emotion"""
        tools_for_emotion = self.mental_tools.get(primary_emotion, ["breathing_exercise"])
        
        tool_descriptions = {
            "breathing_exercise": {
                "name": "4-7-8 Breathing",
                "description": "Breathe in for 4, hold for 7, exhale for 8. Repeat 4 times.",
                "instruction": "Find a comfortable position. Inhale through your nose for 4 counts, hold your breath for 7 counts, then exhale completely through your mouth for 8 counts."
            },
            "grounding_5_4_3_2_1": {
                "name": "5-4-3-2-1 Grounding",
                "description": "Name 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste.",
                "instruction": "Look around and name: 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste."
            },
            "gratitude_practice": {
                "name": "Gratitude Reflection",
                "description": "Think of 3 things you're grateful for today, no matter how small.",
                "instruction": "Take a moment to think of three things you're grateful for today. They can be big or small - maybe the warmth of your coffee, a kind text, or simply that you made it through today."
            },
            "self_compassion": {
                "name": "Self-Compassion Exercise",
                "description": "Speak to yourself as you would to a dear friend.",
                "instruction": "Place your hand on your heart and say: 'This is a moment of suffering. Suffering is part of life. May I be kind to myself in this moment.'"
            },
            "progressive_muscle_relaxation": {
                "name": "Progressive Muscle Relaxation",
                "description": "Tense and release muscle groups to reduce physical tension.",
                "instruction": "Starting with your toes, tense each muscle group for 5 seconds, then release. Work your way up through your body."
            }
        }
        
        return [tool_descriptions.get(tool, {"name": tool, "description": "Helpful tool", "instruction": "Practice mindfully"}) 
                for tool in tools_for_emotion[:3]]
    
    def generate_contextual_response(self, transcript: str, emotion_data: Dict, prompt_template: str) -> str:
        """Generate AI response with full context awareness"""
        from prompt import get_crisis_prompt, get_bullying_support_prompt
        
        # Handle crisis situations first
        if emotion_data.get("crisis_detected"):
            crisis_prompt = get_crisis_prompt()
            formatted_prompt = crisis_prompt.format(
                journal_text=transcript,
                emotion=emotion_data.get("primary_emotion", "distressed"),
                intensity=emotion_data.get("intensity", "high")
            )
        elif emotion_data.get("bullying_detected"):
            bullying_prompt = get_bullying_support_prompt()
            formatted_prompt = bullying_prompt.format(
                journal_text=transcript,
                emotion=emotion_data.get("primary_emotion", "hurt"),
                intensity=emotion_data.get("intensity", "high")
            )
        else:
            # Regular response with context
            formatted_prompt = prompt_template.format(
                journal_text=transcript,
                emotion=emotion_data.get("primary_emotion", "neutral"),
                secondary_emotions=", ".join(emotion_data.get("secondary_emotions", [])),
                intensity=emotion_data.get("intensity", "medium"),
                tone=emotion_data.get("tone", "neutral"),
                emotional_summary=emotion_data.get("emotional_summary", "")
            )
        
        # Add conversation history for context
        if self.conversation_history:
            recent_context = ""
            for entry in self.conversation_history[-3:]:  # Last 3 conversations
                recent_context += f"\nPrevious: User felt {entry['emotion_data'].get('primary_emotion', 'neutral')} - {entry['user_input'][:100]}..."
            formatted_prompt += f"\n\nConversation Context:{recent_context}"
        
        # Add user profile context
        if self.user_profile:
            profile_context = f"\nUser Profile: {json.dumps(self.user_profile)}"
            formatted_prompt += profile_context
        
        try:
            # System message based on support style
            support_style = self.user_profile.get("support_style", "mental_health")
            
            if support_style == "spiritual":
                system_message = "You are VoiceMind, a compassionate spiritual guide and faith-based wellness coach. Offer gentle wisdom, hope, and spiritual comfort."
            else:
                system_message = "You are VoiceMind, a professional mental health coach with deep empathy. Provide evidence-based support and practical guidance."
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.7,
                max_tokens=350
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Add mental health tools suggestion if appropriate
            if emotion_data.get("needs_support") and not emotion_data.get("crisis_detected"):
                tools = self.get_mental_health_tools(emotion_data.get("primary_emotion", "neutral"))
                if tools:
                    tool_suggestion = f"\n\n💡 **Helpful Tool for You:**\n**{tools[0]['name']}** - {tools[0]['instruction']}"
                    ai_response += tool_suggestion
            
            return ai_response
            
        except Exception as e:
            return f"I'm here to support you, but I'm having technical difficulties right now. Please know that your feelings are valid and you're not alone. Error: {str(e)}"
    
    def text_to_speech(self, text: str, output_path: str = "response.mp3", lang: str = "en") -> str:
        """Convert text to speech using gTTS"""
        try:
            # Remove any tool suggestions from TTS (they're visual)
            clean_text = re.sub(r'\n\n💡.*', '', text, flags=re.DOTALL)
            clean_text = re.sub(r'\*\*.*?\*\*', '', clean_text)  # Remove bold formatting
            
            tts = gTTS(text=clean_text, lang=lang, slow=False)
            tts.save(output_path)
            return output_path
        except Exception as e:
            raise Exception(f"Text-to-speech conversion failed: {str(e)}")
    
    def play_audio(self, audio_path: str) -> None:
        """Play audio file using pygame"""
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
                
        except Exception as e:
            print(f"Audio playback failed: {str(e)}")
    
    def process_voice_conversation(self, audio_path: str, support_style: str = "mental_health") -> Dict[str, Any]:
        """
        Complete voice-to-voice conversation processing pipeline
        Main method for backend integration
        """
        from prompt import get_prompt_by_style
        
        try:
            # Step 1: Transcribe audio
            transcript = self.transcribe_audio(audio_path)
            
            # Step 2: Analyze emotions and detect crisis
            emotion_data = self.analyze_emotion_and_crisis(transcript)
            
            # Step 3: Get appropriate prompt based on support style and situation
            prompt_template = get_prompt_by_style(support_style, emotion_data)
            
            # Step 4: Generate contextual AI response
            ai_response = self.generate_contextual_response(transcript, emotion_data, prompt_template)
            
            # Step 5: Convert response to speech
            response_audio_path = f"voicemind_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            self.text_to_speech(ai_response, response_audio_path)
            
            # Step 6: Add to conversation history
            self.add_to_conversation_history(transcript, ai_response, emotion_data)
            
            # Step 7: Get mental health tools if needed
            mental_tools = []
            if emotion_data.get("needs_support") and not emotion_data.get("crisis_detected"):
                mental_tools = self.get_mental_health_tools(emotion_data.get("primary_emotion", "neutral"))
            
            return {
                "success": True,
                "transcript": transcript,
                "emotion_analysis": emotion_data,
                "ai_response": ai_response,
                "response_audio_path": response_audio_path,
                "mental_tools": mental_tools,
                "crisis_detected": emotion_data.get("crisis_detected", False),
                "bullying_detected": emotion_data.get("bullying_detected", False),
                "conversation_id": self.conversation_history[-1]["session_id"] if self.conversation_history else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "crisis_detected": False,
                "bullying_detected": False
            }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation history for dashboard"""
        if not self.conversation_history:
            return {"total_sessions": 0, "dominant_emotions": [], "progress_indicators": []}
        
        emotions = [entry["emotion_data"].get("primary_emotion", "neutral") for entry in self.conversation_history]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            "total_sessions": len(self.conversation_history),
            "dominant_emotions": sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "recent_mood_trend": emotions[-5:] if len(emotions) >= 5 else emotions,
            "crisis_incidents": sum(1 for entry in self.conversation_history if entry["emotion_data"].get("crisis_detected", False)),
            "last_session": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }

# Factory function for easy backend integration
def create_voicemind_instance(api_key: str, user_profile: Optional[Dict] = None) -> VoiceMindAI:
    """Factory function to create VoiceMind AI instance"""
    ai = VoiceMindAI(api_key)
    if user_profile:
        ai.set_user_profile(user_profile)
    return ai

# Live conversation wrapper for backend integration
def start_live_voice_chat(api_key: str, user_profile: Dict, support_style: str = "mental_health") -> Dict[str, Any]:
    """
    One-line function for backend to start live voice conversation
    
    Args:
        api_key (str): OpenAI API key
        user_profile (Dict): User information
        support_style (str): "mental_health" or "spiritual"
        
    Returns:
        Dict: Complete conversation result
    """
    ai = create_voicemind_instance(api_key, user_profile)
    return ai.start_live_conversation(support_style)