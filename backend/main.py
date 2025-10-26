from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import os
from dotenv import load_dotenv
from typing import List, Optional, Set, Dict
import json
import io
import joblib
import numpy as np
from mbti import get_mbti_communication_style
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="AI Conversation Assistant", version="1.0.0")

# In-memory storage for user settings (in production, use a database)
user_settings: Dict[str, Dict] = {}

# Default voice settings (can be customized per user)
DEFAULT_VOICE_SETTINGS = {
    "stability": 0.5,              # 0.0=very expressive, 1.0=very consistent
    "similarity_boost": 0.75,      # How closely to match original voice (0.0-1.0)
    "style": 0.0,                  # Style exaggeration (0.0=neutral, 1.0=max, adds latency)
    "use_speaker_boost": True      # Speaker enhancement (recommended: True)
}

# Pydantic models for request/response
class VoiceSelection(BaseModel):
    user_id: str
    voice_id: str
    voice_name: Optional[str] = None

class VoiceSettings(BaseModel):
    user_id: str
    stability: float
    similarity_boost: float
    style: float
    use_speaker_boost: bool

# Load emotion classification model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model_binary.joblib")
emotion_model = None
emotion_scaler = None
label_map = None

try:
    model_data = joblib.load(MODEL_PATH)

    # Extract model, scaler, and label map from the dictionary
    emotion_model = model_data['model']
    emotion_scaler = model_data['scaler']
    label_map = model_data.get('label_map', {})

    # Reverse the label map (it's currently {'happy': '0', 'sadness': '1'})
    # We need {0: 'happy', 1: 'sadness'}
    reverse_label_map = {int(v): k for k, v in label_map.items()}

    print(f"✓ Emotion model loaded successfully")
    print(f"  Label mapping: {reverse_label_map}")
except Exception as e:
    print(f"✗ Failed to load emotion model: {e}")
    emotion_model = None
    emotion_scaler = None
    reverse_label_map = {0: 'happy', 1: 'sadness'}

def predict_emotion(features):
    """Simple function to predict emotion from features"""
    if emotion_model is None or emotion_scaler is None:
        return None, 0.0

    # Scale features
    features_array = np.array(features).reshape(1, -1)
    features_scaled = emotion_scaler.transform(features_array)

    # Predict
    prediction = emotion_model.predict(features_scaled)[0]
    proba = emotion_model.predict_proba(features_scaled)[0]
    confidence = float(np.max(proba))

    # Use the actual label map from the model
    emotion_label = reverse_label_map.get(prediction, 'unknown')

    # Normalize 'sadness' to 'sad' for frontend
    emotion = 'sad' if emotion_label == 'sadness' else emotion_label

    print(f"  [DEBUG] Raw prediction: {prediction} → {emotion_label} → {emotion}")

    return emotion, confidence

# WebSocket connection manager for broadcasting emotions
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.discard(conn)

manager = ConnectionManager()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
LAVA_BASE_URL = os.getenv("LAVA_BASE_URL")
LAVA_FORWARD_TOKEN = os.getenv("LAVA_FORWARD_TOKEN")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "AI Conversation Assistant API"}

@app.post("/api/user/voice")
async def set_user_voice(selection: VoiceSelection):
    """Set the user's selected voice ID"""
    user_settings[selection.user_id] = {
        "voice_id": selection.voice_id,
        "voice_name": selection.voice_name
    }
    print(f"✓ User {selection.user_id} selected voice: {selection.voice_name} ({selection.voice_id})")
    return {"status": "success", "message": f"Voice set to {selection.voice_name}"}

@app.get("/api/user/{user_id}/voice")
async def get_user_voice(user_id: str):
    """Get the user's selected voice ID"""
    if user_id in user_settings and "voice_id" in user_settings[user_id]:
        return {
            "voice_id": user_settings[user_id]["voice_id"],
            "voice_name": user_settings[user_id].get("voice_name")
        }
    # Return default fallback
    return {
        "voice_id": ELEVENLABS_VOICE_ID,
        "voice_name": "Default"
    }

@app.post("/api/user/voice-settings")
async def set_user_voice_settings(settings: VoiceSettings):
    """Save user's custom voice settings"""
    if settings.user_id not in user_settings:
        user_settings[settings.user_id] = {}

    user_settings[settings.user_id]["voice_settings"] = {
        "stability": settings.stability,
        "similarity_boost": settings.similarity_boost,
        "style": settings.style,
        "use_speaker_boost": settings.use_speaker_boost
    }
    print(f"✓ User {settings.user_id} updated voice settings: stability={settings.stability}, similarity={settings.similarity_boost}, style={settings.style}")
    return {"status": "success", "message": "Voice settings saved"}

@app.get("/api/user/{user_id}/voice-settings")
async def get_user_voice_settings(user_id: str):
    """Get user's custom voice settings or return defaults"""
    if user_id in user_settings and "voice_settings" in user_settings[user_id]:
        return user_settings[user_id]["voice_settings"]
    return DEFAULT_VOICE_SETTINGS

@app.post("/api/speech-to-text")
async def speech_to_text(audio_file: UploadFile = File(...)):
    """Convert audio to text using OpenAI Whisper via Lava Payments"""
    try:
        # Read audio file
        audio_data = await audio_file.read()
        
        # Prepare request to OpenAI via Lava
        url = f"{LAVA_BASE_URL}/forward?u=https://api.openai.com/v1/audio/transcriptions"
        
        headers = {
            "Authorization": f"Bearer {LAVA_FORWARD_TOKEN}",
        }
        
        files = {
            "file": ("audio.webm", io.BytesIO(audio_data), "audio/webm")
        }
        data = {
            "model": "whisper-1",
            "language": "en"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            text = result.get("text") or result.get("response", {}).get("text")
            return {"text": text or "No transcription returned."}
            
    except httpx.HTTPStatusError as e:
        detail = f"Lava/OpenAI API error {e.response.status_code}: {e.response.text}"
        print(f"❌ Speech-to-text error: {detail}")
        raise HTTPException(status_code=500, detail=detail)
    except Exception as e:
        detail = f"Speech-to-text error: {str(e)}"
        print(f"❌ {detail}")
        raise HTTPException(status_code=500, detail=detail)

@app.post("/api/generate-responses")
async def generate_responses(request: Request):
    """Generate four responses with different energy levels using OpenAI via Lava"""
    try:
        # Parse JSON request body
        body = await request.json()
        user_keywords = body.get("user_keywords", "")
        previous_conversation = body.get("previous_conversation", "")
        emotional_state = body.get("emotional_state", "neutral")
        personality_type = body.get("personality_type", "")
        
        # Calculate keyword count for dynamic length guidance
        keyword_count = len([k.strip() for k in user_keywords.split() if k.strip()])
        
        # Dynamic length guidance based on keywords
        if keyword_count <= 3:
            length_guidance = {
                "low": "3-7 words",
                "medium": "1 sentence (8-15 words)",
                "high": "2 sentences (15-25 words)",
                "contradictory": "1 sentence (8-15 words)"
            }
        elif keyword_count <= 6:
            length_guidance = {
                "low": "5-10 words",
                "medium": "1-2 sentences (15-25 words)",
                "high": "2-3 sentences (25-40 words)",
                "contradictory": "1-2 sentences (15-25 words)"
            }
        else:  # 7+ keywords
            length_guidance = {
                "low": "8-12 words",
                "medium": "2 sentences (20-35 words)",
                "high": "3-4 sentences (40-60 words)",
                "contradictory": "2 sentences (20-35 words)"
            }
        
        # Build personality context with MBTI function
        personality_context = ""
        if personality_type:
            mbti_data = get_mbti_communication_style(personality_type)
            personality_context = f"""
                Personality Type: {personality_type}
                Communication Style: {mbti_data['communication_style']}
                Preferred Vocabulary: {mbti_data['vocabulary_preferences']}

                Few-Shot Examples for {personality_type}:
                - Low energy: "{mbti_data['few_shot_examples']['low']}"
                - Medium energy: "{mbti_data['few_shot_examples']['medium']}"
                - High energy: "{mbti_data['few_shot_examples']['high']}"
                - Contradictory: "{mbti_data['few_shot_examples']['contradictory']}"

                Match this communication pattern in your generated responses."""
                        
        prompt = f"""
                You are a sentence builder helping the user express themselves naturally.

                The user has given you {keyword_count} keywords or phrases. Some may be grammatically incomplete (missing words like "am", "is", "the", etc.).

                Keywords: {user_keywords}
                User's emotion: {emotional_state}{personality_context}

                YOUR TASK:
                - Build natural sentences that capture the MEANING of these keywords
                - Fix grammar by adding necessary words ("I great" → "I'm great" or "I feel great")
                - The user will SPEAK these - make them sound natural and complete
                - DO NOT just copy-paste broken phrases

                Build 4 variations with different energy:

                "low" ({length_guidance['low']}):
                - Brief and understated
                - {emotional_state}: {'calm' if emotional_state == 'happy' else 'subdued'}

                "medium" ({length_guidance['medium']}):
                - Natural sentence length
                - {emotional_state}: {'genuine positivity' if emotional_state == 'happy' else 'sincere concern'}

                "high" ({length_guidance['high']}):
                - Expressive and detailed
                - {emotional_state}: {'enthusiastic' if emotional_state == 'happy' else 'deeply emotional'}

                "contradictory" ({length_guidance['contradictory']}):
                - Sarcastic/ironic
                - {emotional_state}: {'ironic understatement' if emotional_state == 'happy' else 'dark humor'}

                Output format: {{"low": "...", "medium": "...", "high": "...", "contradictory": "..."}}

                EXAMPLES:

                Keywords: "sound good, i great too"
                Emotion: neutral
                Output:
                {{"low": "Sounds good, I'm great.", "medium": "That sounds good to me, and honestly I'm feeling great too.", "high": "That sounds really good! I'm actually feeling great about this too - everything seems to be going well.", "contradictory": "Oh yeah, sounds absolutely wonderful. I'm just great. Living the dream."}}

                Keywords: "tired work late"
                Emotion: sad
                Output:
                {{"low": "Tired from work.", "medium": "I'm really tired because I've been working late.", "high": "I'm exhausted from working so late every night - it's really draining and I feel completely worn out.", "contradictory": "Yeah, working late is just fantastic. Loving how tired I am."}}

                Keywords: "meeting deadline project"
                Emotion: happy
                Output:
                {{"low": "Met the deadline!", "medium": "We met the project deadline for the meeting!", "high": "We actually met the project deadline! I'm so excited for this meeting - everything came together perfectly!", "contradictory": "Oh yeah, we met the deadline. Not stressful at all."}}

                Now build sentences using: {user_keywords}
                Emotion: {emotional_state}

                Output ONLY JSON:
                """

        url = f"{LAVA_BASE_URL}/forward?u=https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LAVA_FORWARD_TOKEN}",
        }
        
        payload = {
            "model": "gpt-5-chat-latest",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.35,
            "max_tokens": 1000  # Increased for flexibility with more keywords
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse JSON response
            try:
                responses = json.loads(content)
                return {
                    "low": responses.get("low", "I understand."),
                    "medium": responses.get("medium", "That makes sense."),
                    "high": responses.get("high", "That sounds great!"),
                    "contradictory": responses.get("contradictory", "Oh wonderful.")
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                print(f"⚠️ JSON parsing failed, content: {content}")
                print(f"📊 Keyword count was: {keyword_count}")
                return {
                    "low": "I understand.",
                    "medium": "That makes sense.",
                    "high": "That sounds great!",
                    "contradictory": "Oh wonderful."
                }
            
    except httpx.HTTPStatusError as e:
        error_detail = f"Lava API error {e.response.status_code}: {e.response.text}"
        print(f"❌ Generate responses error: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)
    except Exception as e:
        error_detail = f"Response generation failed: {str(e)}"
        print(f"❌ Generate responses error: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

def apply_energy_range(base_value: float, energy: str) -> float:
    """Apply energy-based range to a setting value

    Ranges:
    - low: 0.0-0.33
    - medium: 0.34-0.66
    - high: 0.67-1.0
    - contradictory: 0.1 (fixed)
    """
    if energy == "low":
        # Map base_value (0-1) to range 0.0-0.33
        return base_value * 0.33
    elif energy == "medium":
        # Map base_value (0-1) to range 0.34-0.66
        return 0.34 + (base_value * 0.32)
    elif energy == "high":
        # Map base_value (0-1) to range 0.67-1.0
        return 0.67 + (base_value * 0.33)
    elif energy == "contradictory":
        # Fixed value for sarcasm
        return 0.1
    else:
        # Default to medium range
        return 0.34 + (base_value * 0.32)

@app.post("/api/text-to-speech")
async def text_to_speech(
    text: str = Form(...),
    energy: str = Form("medium"),                    # low/medium/high/contradictory
    emotional_state: str = Form("neutral"),          # happy/sad from EEG
    user_id: str = Form("default_user")              # user ID to get their voice preference
):
    """Convert text to speech using ElevenLabs with dynamic voice settings based on energy level and emotion"""
    try:
        # Get user's base voice settings or defaults
        base_settings = DEFAULT_VOICE_SETTINGS.copy()
        if user_id in user_settings and "voice_settings" in user_settings[user_id]:
            base_settings = user_settings[user_id]["voice_settings"].copy()
            print(f"🎨 Using {user_id}'s custom voice settings")
        else:
            print(f"🎨 Using default voice settings")

        # Apply energy-based range to stability (contradictory uses fixed value)
        if energy == "contradictory":
            # For sarcasm, use fixed low values
            settings = {
                "stability": 0.1,
                "similarity_boost": base_settings["similarity_boost"],
                "style": 0.1,
                "use_speaker_boost": base_settings["use_speaker_boost"]
            }
        else:
            # Apply energy range to the user's base stability setting
            settings = {
                "stability": apply_energy_range(base_settings["stability"], energy),
                "similarity_boost": base_settings["similarity_boost"],
                "style": apply_energy_range(base_settings["style"], energy),
                "use_speaker_boost": base_settings["use_speaker_boost"]
            }

        # Get user's selected voice ID or use default from env
        voice_id = ELEVENLABS_VOICE_ID  # Default fallback
        if user_id in user_settings and "voice_id" in user_settings[user_id]:
            voice_id = user_settings[user_id]["voice_id"]
            print(f"🎤 Using user's selected voice: {user_settings[user_id].get('voice_name', 'Unknown')} ({voice_id})")
        else:
            print(f"🎤 Using default voice from environment")

        # Log for debugging
        print(f"🎤 TTS Request: energy={energy}, emotion={emotional_state}, user={user_id}")
        print(f"   Applied settings: stability={settings['stability']:.2f}, similarity={settings['similarity_boost']:.2f}, style={settings['style']:.2f}")

        url = f"{LAVA_BASE_URL}/forward?u=https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LAVA_FORWARD_TOKEN}"
        }
        
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": settings
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:  # Added timeout
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            return StreamingResponse(
                io.BytesIO(response.content),
                media_type="audio/mpeg",
                headers={"Content-Disposition": "attachment; filename=speech.mp3"}
            )
            
    except httpx.HTTPStatusError as e:
        error_detail = f"ElevenLabs API error {e.response.status_code}: {e.response.text}"
        print(f"❌ TTS error: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)
    except Exception as e:
        error_detail = f"TTS failed: {str(e)}"
        print(f"❌ TTS error: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for receiving EEG predictions from live_stream.py"""
    await websocket.accept()
    print("✓ EEG stream connected")

    # Send welcome message
    await websocket.send_json({
        "type": "connection",
        "message": "Connected to emotion prediction server"
    })

    try:
        while True:
            # Receive message from live_stream.py
            data = await websocket.receive_json()
            print(f"📥 Received from EEG: {data.get('type')}")

            if data.get("type") == "predict":
                features = data.get("features")
                print(f"🔢 Features received: {len(features) if features else 0}")

                if features and len(features) == 52:
                    try:
                        # Simple: predict emotion
                        emotion, confidence = predict_emotion(features)

                        if emotion:
                            print(f"📊 {emotion.upper()} ({confidence:.0%}) → Broadcasting to {len(manager.active_connections)} frontend(s)")

                            # Send back to EEG stream
                            await websocket.send_json({
                                "type": "prediction",
                                "emotion": emotion,
                                "confidence": confidence
                            })

                            # Broadcast to frontend
                            await manager.broadcast({
                                "type": "emotion_update",
                                "emotion": emotion,
                                "confidence": confidence
                            })
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Model not loaded"
                            })

                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Error: {str(e)}"
                        })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Expected 52 features, got {len(features) if features else 0}"
                    })

    except WebSocketDisconnect:
        print("EEG stream disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.websocket("/ws/emotions")
async def emotion_stream(websocket: WebSocket):
    """WebSocket endpoint for frontend clients to receive emotion updates"""
    await manager.connect(websocket)
    print(f"✓ Frontend client connected (total: {len(manager.active_connections)})")

    # Send initial connection confirmation
    await websocket.send_json({
        "type": "connected",
        "message": "Connected to emotion stream"
    })

    try:
        # Keep connection alive and listen for any client messages (like pings)
        while True:
            try:
                message = await websocket.receive_text()
                # Echo back ping messages to keep connection alive
                if message == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception as e:
                # If there's an error receiving, the connection might be closed
                break
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"✗ Frontend client disconnected (remaining: {len(manager.active_connections)})")
    except Exception as e:
        manager.disconnect(websocket)
        print(f"✗ Emotion stream error: {e} (remaining: {len(manager.active_connections)})")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)