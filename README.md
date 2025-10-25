# AI Conversation Assistant

A real-time conversation assistant that uses AI to generate contextual responses based on keywords, conversation history, and emotional state from EEG data.

## Features

- **Real-time Audio Recording**: Browser-based microphone recording with Web Audio API
- **Speech-to-Text**: OpenAI Whisper integration via Lava Payments
- **AI Response Generation**: GPT-3.5-turbo for generating contextual responses
- **Text-to-Speech**: ElevenLabs integration with sentiment-based voice modulation
- **Emotional State Integration**: EEG data visualization and processing
- **Multi-sentiment Responses**: Generate calm, neutral, and excited response options

## Tech Stack

- **Frontend**: React + Vite
- **Backend**: FastAPI (Python)
- **AI Services**: OpenAI (GPT-3.5, Whisper) via Lava Payments
- **TTS**: ElevenLabs
- **Audio**: Web Audio API

## Setup Instructions

### 1. Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
python main.py
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### 3. Environment Variables

Create `.env` file in backend directory:

```env
LAVA_BASE_URL=your_lava_base_url
LAVA_FORWARD_TOKEN=your_lava_forward_token
ELEVENLABS_VOICE_ID=your_voice_id
```

## Usage

1. **Start Recording**: Click the microphone button to start recording
2. **Enter Keywords**: Type keywords in the input field
3. **Generate Responses**: Click "Generate" to get AI responses
4. **Select Response**: Choose from calm, neutral, or excited options
5. **Listen**: The selected response will be spoken with appropriate sentiment

## API Endpoints

- `POST /api/speech-to-text` - Convert audio to text
- `POST /api/generate-responses` - Generate AI responses
- `POST /api/text-to-speech` - Convert text to speech with sentiment
- `GET /api/health` - Health check

## Development

- Backend runs on `http://localhost:8000`
- Frontend runs on `http://localhost:3000`
- CORS is configured for development
- Use ngrok for microphone access in production

## Hackathon Notes

- 24-hour development timeline
- Lava Payments integration for OpenAI API calls
- EEG emotional state simulation
- Real-time conversation flow
- Sentiment-based response generation
