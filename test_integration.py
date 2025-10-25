#!/usr/bin/env python3
"""
Integration test for AI Conversation Assistant
Tests the full workflow: audio -> text -> AI response -> TTS
"""

import requests
import json
import time
import os
from pathlib import Path

# Test configuration
BACKEND_URL = "http://localhost:8000"
TEST_AUDIO_FILE = "test_audio.wav"  # You'll need to create this

def test_health_check():
    """Test if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/health")
        if response.status_code == 200:
            print("✅ Backend health check passed")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running. Start it with: python backend/main.py")
        return False

def test_response_generation():
    """Test AI response generation"""
    try:
        test_data = {
            "user_keywords": "excited about the project",
            "previous_conversation": "USER: How are you?\nOTHER: I'm doing great, thanks for asking!",
            "emotional_state": "excited"
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/generate-responses",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Response generation test passed")
            print(f"   Generated responses: {data}")
            return True
        else:
            print(f"❌ Response generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Response generation test failed: {e}")
        return False

def test_text_to_speech():
    """Test text-to-speech functionality"""
    try:
        form_data = {
            "text": "Hello, this is a test of the text to speech system.",
            "sentiment": "neutral"
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/text-to-speech",
            data=form_data
        )
        
        if response.status_code == 200:
            # Save the audio file
            with open("test_output.mp3", "wb") as f:
                f.write(response.content)
            print("✅ Text-to-speech test passed")
            print("   Audio saved as test_output.mp3")
            return True
        else:
            print(f"❌ Text-to-speech failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Text-to-speech test failed: {e}")
        return False

def create_test_audio():
    """Create a simple test audio file using text-to-speech"""
    print("🎵 Creating test audio file...")
    try:
        form_data = {
            "text": "This is a test recording for speech to text conversion.",
            "sentiment": "neutral"
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/text-to-speech",
            data=form_data
        )
        
        if response.status_code == 200:
            with open(TEST_AUDIO_FILE, "wb") as f:
                f.write(response.content)
            print(f"✅ Test audio created: {TEST_AUDIO_FILE}")
            return True
        else:
            print(f"❌ Failed to create test audio: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to create test audio: {e}")
        return False

def test_speech_to_text():
    """Test speech-to-text functionality"""
    if not os.path.exists(TEST_AUDIO_FILE):
        print("⚠️  Test audio file not found. Creating one...")
        if not create_test_audio():
            print("❌ Could not create test audio file")
            return False
    
    try:
        with open(TEST_AUDIO_FILE, "rb") as f:
            files = {"audio_file": f}
            response = requests.post(f"{BACKEND_URL}/api/speech-to-text", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Speech-to-text test passed")
            print(f"   Transcribed text: {data.get('text', 'No text returned')}")
            return True
        else:
            print(f"❌ Speech-to-text failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Speech-to-text test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🧪 AI Conversation Assistant - Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Response Generation", test_response_generation),
        ("Text-to-Speech", test_text_to_speech),
        ("Speech-to-Text", test_speech_to_text),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your AI Conversation Assistant is ready!")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    # Cleanup
    if os.path.exists(TEST_AUDIO_FILE):
        os.remove(TEST_AUDIO_FILE)
    if os.path.exists("test_output.mp3"):
        os.remove("test_output.mp3")

if __name__ == "__main__":
    main()
