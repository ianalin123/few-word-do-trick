from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import os
from dotenv import load_dotenv
from typing import List, Optional, Set
import json
import io
import joblib
import numpy as np

load_dotenv()

app = FastAPI(title="AI Conversation Assistant", version="1.0.0")

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
        print(detail)
        raise HTTPException(status_code=500, detail=detail)

@app.post("/api/generate-responses")
async def generate_responses(request: Request):
    """Generate three responses with different sentiments using OpenAI via Lava"""
    try:
        # Parse JSON request body
        body = await request.json()
        user_keywords = body.get("user_keywords", "")
        previous_conversation = body.get("previous_conversation", "")
        emotional_state = body.get("emotional_state", "neutral")
        
        prompt = f"""
You are an empathetic conversational assistant that helps the user decide what to say next in an ongoing dialogue.

Your goal is to generate three possible user responses that:
1. Are coherent and contextually appropriate given the previous conversation.
2. Use the user_keywords in the responses naturally.
3. Reflect the user's current emotional_state.
4. Differ clearly in tone: calm, neutral, and excited.

Inputs:
user_keywords = '{user_keywords}'
previous_conversation = '''{previous_conversation}'''
emotional_state = '{emotional_state}'

Instructions:
- You must output a valid JSON object with exactly three keys: "calm", "neutral", and "excited".
- Each value must be one complete, coherent sentence the user might say next.
- Use the conversation and emotional_state to guide tone, but keep content logically grounded in the prior context and user keywords.
- Do not include explanations, extra text, or formatting outside the JSON.


Now produce your JSON response:
"""

        url = f"{LAVA_BASE_URL}/forward?u=https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LAVA_FORWARD_TOKEN}",
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 300
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
                    "calm": responses.get("calm", ""),
                    "neutral": responses.get("neutral", ""),
                    "excited": responses.get("excited", "")
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                lines = content.strip().split('\n')
                return {
                    "calm": lines[0] if len(lines) > 0 else "I understand.",
                    "neutral": lines[1] if len(lines) > 1 else "That's interesting.",
                    "excited": lines[2] if len(lines) > 2 else "That sounds great!"
                }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")

@app.post("/api/text-to-speech")
async def text_to_speech(text: str = Form(...), sentiment: str = Form("neutral")):
    """Convert text to speech using ElevenLabs with sentiment"""
    try:
        # Map sentiment to voice settings
        voice_settings = {
            "calm": {"stability": 0.8, "similarity_boost": 0.7},
            "neutral": {"stability": 0.5, "similarity_boost": 0.5},
            "excited": {"stability": 0.3, "similarity_boost": 0.8}
        }
        
        settings = voice_settings.get(sentiment, voice_settings["neutral"])
        
        url = f"{LAVA_BASE_URL}/forward?u=https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        
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
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            return StreamingResponse(
                io.BytesIO(response.content),
                media_type="audio/mpeg",
                headers={"Content-Disposition": "attachment; filename=speech.mp3"}
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

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