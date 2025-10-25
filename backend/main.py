from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import os
from dotenv import load_dotenv
from typing import List, Optional
import json
import io

load_dotenv()

app = FastAPI(title="AI Conversation Assistant", version="1.0.0")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)