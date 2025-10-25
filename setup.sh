#!/bin/bash

# AI Conversation Assistant - Setup Script

echo "🚀 Setting up AI Conversation Assistant..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not installed."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

# Setup backend with virtual environment
echo "📦 Setting up backend with virtual environment..."
cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cat > .env << EOF
LAVA_BASE_URL=your_lava_base_url
LAVA_FORWARD_TOKEN=your_lava_forward_token
ELEVENLABS_VOICE_ID=your_voice_id
EOF
    echo "⚠️  Please edit backend/.env with your actual API keys"
fi

# Setup frontend
echo "📦 Setting up frontend..."
cd ../frontend

echo "📦 Installing Node.js dependencies..."
npm install

echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Edit backend/.env with your API keys"
echo "2. Run ./run.sh to start the application"
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "🔑 Required API keys:"
echo "   - Lava Forward Token (handles OpenAI and ElevenLabs routing)"
echo "   - ElevenLabs Voice ID (for text-to-speech voice selection)"
