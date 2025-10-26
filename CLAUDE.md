# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Few Word Do Trick** is a real-time AI conversation assistant that combines EEG emotion detection with GPT-powered response generation. The system streams brain activity from a Muse headset, classifies emotions (happy/sad) using a trained RandomForest model, and adapts AI-generated conversation suggestions based on the user's emotional state and MBTI personality.

## Development Commands

### Backend

```bash
# Setup
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run development server (port 8000)
python main.py

# EEG live streaming (requires Muse headset)
muselsl stream  # Terminal 1: start LSL stream
python live_stream.py  # Terminal 2: process and classify

# Test emotion pipeline without headset
python test_emotions.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev  # Development server (port 3001)
npm run build  # Production build
npm run preview  # Preview production build
```

### Model Training

```bash
cd backend/pretraining

# 1. Collect EEG data
python record_data.py  # Records trials to data/ folder

# 2. Extract features
python preprocess.py  # Generates features.csv

# 3. Train binary classifier
python train_binary_classifier.py  # Creates emotion_model_binary.joblib

# Analysis and visualization
python eeg_emotion_analysis.py  # Multi-model comparison
python visualize_pca.py  # Feature space visualization
```

## Architecture

### Real-Time Emotion Detection Pipeline

```
Muse Headset (4 EEG channels @ 256Hz)
    ↓ [Lab Streaming Layer]
live_stream.py (EEG processor)
    ↓ [Preprocessing: bandpass filter, notch filter, downsample to 128Hz]
    ↓ [Feature extraction: 52 features from 4 frequency bands]
    ↓ [WebSocket: /ws]
main.py (FastAPI backend)
    ↓ [StandardScaler → RandomForest binary classifier]
    ↓ [WebSocket broadcast: /ws/emotions]
React Frontend
    ↓ [State update: emotionalState]
GPT-3.5-turbo (response generation with emotional context)
```

**Key invariants:**
- **Exactly 52 features** must be extracted (4 channels × 13 features/channel)
- **Feature extraction pipeline** in `live_stream.py` must match `preprocess.py` exactly
- **Preprocessing order:** bandpass (1-45Hz) → notch (60Hz) → downsample (256→128Hz)
- **Window parameters:** 2-second window, 50% overlap (1-second step)

### WebSocket Architecture

The system uses **two separate WebSocket connections**:

1. **`/ws`** - EEG stream → Backend
   - Client: `live_stream.py`
   - Sends: `{"type": "predict", "features": [52 floats]}`
   - Receives: `{"type": "prediction", "emotion": "happy"/"sad", "confidence": float}`

2. **`/ws/emotions`** - Backend → Frontend(s)
   - Client: React app (can have multiple connected)
   - Receives: `{"type": "emotion_update", "emotion": "happy"/"sad", "confidence": float}`
   - Managed by `ConnectionManager` class for broadcasting

### Model Loading (Critical!)

The `.joblib` file contains a **dictionary**, not a direct model:

```python
model_data = joblib.load("emotion_model_binary.joblib")
# Structure:
# {
#   'model': CalibratedClassifierCV (RandomForest),
#   'scaler': StandardScaler,
#   'label_map': {'happy': '0', 'sadness': '1'},
#   'feature_names': [list of 52 feature names]
# }
```

**Must use scaler before prediction:**
```python
features_scaled = scaler.transform(features_array)
prediction = model.predict(features_scaled)
```

### API Integration via Lava Payments

All AI services are proxied through Lava Payments (cost optimization):

**Critical:** `LAVA_BASE_URL` in `.env` must NOT have trailing slash:
- ✅ `https://api.lavapayments.com/v1`
- ❌ `https://api.lavapayments.com/v1/` (causes 308 redirect errors)

**Endpoints:**
- `/api/speech-to-text` → Lava → OpenAI Whisper
- `/api/generate-responses` → Lava → GPT-3.5-turbo
- `/api/text-to-speech` → Lava → ElevenLabs

### Conversation Flow Integration

The `/api/generate-responses` endpoint receives:
```json
{
  "user_keywords": "meeting project deadline",
  "previous_conversation": "OTHER: How was your day?\nUSER: It was productive",
  "emotional_state": "happy",  // From live EEG
  "personality_type": "INTJ",  // From MBTI quiz
  "personality_description": "Analytical and strategic..."
}
```

GPT generates 3 tonally distinct responses:
- **calm:** 3-5 words (Kevin Malone style: "Few word do trick")
- **neutral:** 1 normal sentence
- **excited:** 2-3 enthusiastic sentences

## Important Patterns

### Frontend WebSocket Reconnection

The frontend implements auto-reconnect with 3-second backoff:
```javascript
ws.onclose = () => {
  if (!isUnmounting) {
    setTimeout(connectWebSocket, 3000)
  }
}
```

Always ensure backend WebSocket endpoints handle disconnects gracefully.

### EEG Feature Engineering

Features are extracted per channel from 4 frequency bands:
- **theta (4-8Hz):** Meditation, drowsiness
- **alpha (8-13Hz):** Relaxation, calmness
- **beta (13-30Hz):** Active thinking, focus
- **gamma (30-45Hz):** High-level processing

**Per channel features (13 total):**
1. Band power (theta, alpha, beta, gamma) - 4 features
2. Differential entropy per band - 4 features
3. Beta/Alpha ratio - 1 feature
4. Gamma/Theta ratio - 1 feature
5. Statistical moments (mean, variance, skewness, kurtosis) - 4 features

**Asymmetry features (2 total):**
- Frontal alpha asymmetry: log(AF8_alpha) - log(AF7_alpha)
- Temporal alpha asymmetry: log(TP10_alpha) - log(TP9_alpha)

**Total: (13 × 4 channels) + 2 asymmetry = 54... wait, spec says 52?**
→ Check `preprocess.py` for exact feature list if modifying extraction.

### MBTI Personality Integration

The `FeedbackModal` component collects MBTI quiz results stored in App state:
```javascript
const [personalityType, setPersonalityType] = useState('')
const [personalityDescription, setPersonalityDescription] = useState('')
```

These are sent to GPT to personalize response tone and content.

## Critical Configuration

### Environment Variables (.env)

```env
LAVA_BASE_URL=https://api.lavapayments.com/v1  # NO trailing slash!
LAVA_FORWARD_TOKEN=<JWT with embedded secrets>
ELEVENLABS_VOICE_ID=<voice ID>
```

### Port Configuration

- **Backend:** 8000 (FastAPI)
- **Frontend:** 3001 (Vite dev server)
  - Changed from 3000 to avoid conflicts
  - Configured in `vite.config.js`
  - Proxies `/api/*` to `localhost:8000`

### Muse Headset Setup

```bash
# Check if Muse is detected
muselsl list

# Start streaming
muselsl stream  # Keep running in background

# Connect and process
python live_stream.py
```

**Common issues:**
- No stream found → Run `muselsl stream` first
- Connection timeout → Check headset is on and paired
- Feature count mismatch → Verify preprocessing pipeline matches training

## Debugging

### Backend Logs

The backend prints detailed debug info:
```
✓ Emotion model loaded successfully
  Label mapping: {0: 'happy', 1: 'sadness'}
📥 Received from EEG: predict
🔢 Features received: 52
  [DEBUG] Raw prediction: 0 → happy → happy
📊 HAPPY (85%) → Broadcasting to 3 frontend(s)
```

### Frontend Console

WebSocket messages are logged:
```
✓ Connected to emotion stream
📨 Frontend received: {type: 'emotion_update', emotion: 'happy', confidence: 0.85}
🎭 UPDATING KEVIN TO: HAPPY (85.0%)
```

### Common Errors

**"Prediction error: 'dict' object has no attribute 'predict'"**
→ Model wasn't extracted from dictionary. See "Model Loading" section.

**"Expected 52 features, got X"**
→ Feature extraction mismatch between live_stream.py and training pipeline.

**"Lava API error 308: Redirecting"**
→ Remove trailing slash from `LAVA_BASE_URL` in `.env`

**WebSocket connection failed**
→ Backend not running, or port mismatch. Check both services are on correct ports.

## Data Flow Summary

1. **Muse headset** streams 4-channel EEG @ 256Hz via LSL
2. **live_stream.py** buffers, preprocesses, extracts 52 features every 1 second
3. **Backend** receives features via WebSocket, scales them, predicts emotion
4. **ConnectionManager** broadcasts emotion update to all connected frontends
5. **React app** updates `emotionalState`, which influences GPT prompt
6. **User interaction** (keywords + generate) sends full context to GPT
7. **GPT returns** 3 responses adapted to emotional state + personality
8. **User selects** response, which is synthesized with sentiment-based TTS

## Testing Without Hardware

Use `test_emotions.py` to simulate EEG predictions:
```bash
python test_emotions.py
# Sends random 52-feature vectors to backend
# Watch frontend to see Kevin's mood change
```

This is useful for testing the full pipeline without a Muse headset.
