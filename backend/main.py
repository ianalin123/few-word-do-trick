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
from services.firestore_service import init_firestore, create_conversation, append_message
from services.speaker_service import speaker_service
from models.firestore_models import MessageCreate
from agents.conversation_agent import conversation_graph
from datetime import datetime, timezone

load_dotenv()

_FIREBASE_KEY_PATH = os.path.join(os.path.dirname(__file__), "..", "firebase-admin-key.json")
init_firestore(os.path.abspath(_FIREBASE_KEY_PATH))

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
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


class ConversationCreateRequest(BaseModel):
    user_id: str


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "AI Conversation Assistant API"}


@app.post("/api/conversations")
async def create_conversation_endpoint(req: ConversationCreateRequest):
    conv_id = create_conversation(req.user_id)
    return {"conv_id": conv_id}


@app.post("/api/conversations/{conv_id}/messages")
async def append_message_endpoint(conv_id: str, message: MessageCreate):
    msg_id = append_message(conv_id, message)
    return {"msg_id": msg_id}

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
async def speech_to_text(
    audio_file: UploadFile = File(...),
    conv_id: Optional[str] = Form(None),
):
    """Convert audio to text using OpenAI Whisper via Lava Payments, then diarize and persist."""
    try:
        audio_data = await audio_file.read()

        url = f"{LAVA_BASE_URL}/forward?u=https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {LAVA_FORWARD_TOKEN}"}
        files = {"file": ("audio.webm", io.BytesIO(audio_data), "audio/webm")}
        data = {"model": "whisper-1", "language": "en"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, files=files, data=data)
            print(f"🎙️ Lava STT response: {response.status_code} {response.text[:200]}")
            response.raise_for_status()
            result = response.json()
            text = result.get("text") or result.get("response", {}).get("text") or "No transcription returned."

        speaker_id = "speaker_unknown"
        persisted = False
        if conv_id:
            speaker_id, _sim = speaker_service.identify_speaker(conv_id, audio_data)
            try:
                append_message(
                    conv_id,
                    MessageCreate(
                        speaker=speaker_id,
                        text=text,
                        timestamp=datetime.now(timezone.utc),
                        emotion=None,
                    ),
                )
                persisted = True
            except Exception as e:
                print(f"⚠️ append_message failed: {e}")

        return {"text": text, "speaker_id": speaker_id, "persisted": persisted}

    except httpx.HTTPStatusError as e:
        detail = f"Lava/OpenAI API error {e.response.status_code}: {e.response.text}"
        print(f"❌ Speech-to-text error: {detail}")
        raise HTTPException(status_code=500, detail=detail)
    except Exception as e:
        import traceback
        print(f"❌ Speech-to-text error: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-responses")
async def generate_responses(request: Request):
    """Generate four responses using the LangGraph conversation agent."""
    import asyncio
    try:
        body = await request.json()
        personality_type = body.get("personality_type", "")
        print(f"🧠 Personality context: {'injected (' + personality_type + ')' if personality_type else 'none'}")

        initial_state = {
            "transcript": body.get("user_keywords", ""),
            "emotion": body.get("emotional_state", "neutral"),
            "confidence": 0.0,
            "speaker_id": "USER",
            "conversation_history": [],
            "candidate_response": "",
            "audio_url": "",
            "conv_id": body.get("conv_id", ""),
            "user_id": body.get("user_id", ""),
            "personality_type": personality_type,
            "personality_description": body.get("personality_description", ""),
            "all_responses": {},
        }

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: conversation_graph.invoke(initial_state)
        )
        return result["all_responses"]

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
            settings = {
                "stability": 0.1,
                "similarity_boost": base_settings["similarity_boost"],
                "style": 0.1,
                "use_speaker_boost": base_settings["use_speaker_boost"]
            }
        else:
            settings = {
                "stability": apply_energy_range(base_settings["stability"], energy),
                "similarity_boost": base_settings["similarity_boost"],
                "style": apply_energy_range(base_settings["style"], energy),
                "use_speaker_boost": base_settings["use_speaker_boost"]
            }

        # Apply emotion offset on top of energy settings
        # happy → more expressive (lower stability, higher style)
        # sad   → more monotone  (higher stability, lower style)
        emotion_offsets = {
            "happy": {"stability": -0.2, "style": +0.2},
            "sad":   {"stability": +0.2, "style": -0.2},
        }
        if emotional_state in emotion_offsets:
            offset = emotion_offsets[emotional_state]
            settings["stability"] = max(0.0, min(1.0, settings["stability"] + offset["stability"]))
            settings["style"]     = max(0.0, min(1.0, settings["style"]     + offset["style"]))

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

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
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

            if data.get("type") == "predict":
                features = data.get("features")

                if features and len(features) == 52:
                    try:
                        # Simple: predict emotion
                        emotion, confidence = predict_emotion(features)

                        if emotion:
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