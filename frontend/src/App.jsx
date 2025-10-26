import React, { useState, useEffect } from 'react'
import AudioRecorder from './components/AudioRecorder'
import ConversationDisplay from './components/ConversationDisplay'
import ResponseSelector from './components/ResponseSelector'
import EmotionalStateDisplay from './components/EmotionalStateDisplay'
import KeywordInput from './components/KeywordInput'
import './App.css'

function App() {
  const [conversation, setConversation] = useState([])
  const [emotionalState, setEmotionalState] = useState('neutral')
  const [userKeywords, setUserKeywords] = useState('')
  const [generatedResponses, setGeneratedResponses] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)

  // Connect to WebSocket for live emotion updates from EEG
  useEffect(() => {
    let ws = null
    let reconnectTimer = null
    let isUnmounting = false

    const connectWebSocket = () => {
      if (isUnmounting) return

      try {
        ws = new WebSocket('ws://localhost:8000/ws/emotions')

        ws.onopen = () => {
          console.log('✓ Connected to emotion stream')
          // Send a ping to keep connection alive
          ws.send('ping')
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            console.log('📨 Frontend received:', data)

            if (data.type === 'emotion_update') {
              console.log(`🎭 UPDATING KEVIN TO: ${data.emotion.toUpperCase()} (${(data.confidence * 100).toFixed(1)}%)`)
              setEmotionalState(data.emotion)
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error)
          }
        }

        ws.onerror = (error) => {
          console.error('WebSocket error - is backend running?', error)
        }

        ws.onclose = () => {
          console.log('Disconnected from emotion stream')

          // Attempt to reconnect after 3 seconds if not unmounting
          if (!isUnmounting) {
            console.log('Will attempt to reconnect in 3 seconds...')
            reconnectTimer = setTimeout(connectWebSocket, 3000)
          }
        }
      } catch (error) {
        console.error('Failed to create WebSocket:', error)
        if (!isUnmounting) {
          reconnectTimer = setTimeout(connectWebSocket, 3000)
        }
      }
    }

    // Initial connection
    connectWebSocket()

    // Cleanup on unmount
    return () => {
      isUnmounting = true
      if (reconnectTimer) {
        clearTimeout(reconnectTimer)
      }
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close()
      }
    }
  }, [])

  const handleAudioTranscription = (transcription) => {
    const newMessage = {
      id: Date.now(),
      speaker: 'OTHER',
      text: transcription,
      timestamp: new Date().toLocaleTimeString()
    }
    setConversation(prev => [...prev, newMessage])
  }

  const handleGenerateResponses = async () => {
    if (!userKeywords.trim()) {
      alert('Please enter some keywords')
      return
    }

    setIsProcessing(true)
    try {
      const conversationText = conversation
        .map(msg => `${msg.speaker}: ${msg.text}`)
        .join('\n')

      const response = await fetch('/api/generate-responses', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_keywords: userKeywords,
          previous_conversation: conversationText,
          emotional_state: emotionalState
        })
      })

      if (!response.ok) {
        throw new Error('Failed to generate responses')
      }

      const data = await response.json()
      setGeneratedResponses(data)
    } catch (error) {
      console.error('Error generating responses:', error)
      alert('Failed to generate responses. Please try again.')
    } finally {
      setIsProcessing(false)
    }
  }

  const handleResponseSelect = async (response, sentiment) => {
    // Add user's selected response to conversation
    const userMessage = {
      id: Date.now(),
      speaker: 'USER',
      text: response,
      timestamp: new Date().toLocaleTimeString()
    }
    setConversation(prev => [...prev, userMessage])

    // Play text-to-speech
    try {
      const formData = new FormData()
      formData.append('text', response)
      formData.append('sentiment', sentiment)

      const response_audio = await fetch('/api/text-to-speech', {
        method: 'POST',
        body: formData
      })

      if (response_audio.ok) {
        const audioBlob = await response_audio.blob()
        const audioUrl = URL.createObjectURL(audioBlob)
        const audio = new Audio(audioUrl)
        audio.play()
      }
    } catch (error) {
      console.error('Error playing audio:', error)
    }

    // Clear generated responses
    setGeneratedResponses(null)
    setUserKeywords('')
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Why waste time say lot word when few word do trick?</h1>
        <p>Real-time conversation assistant with AI-powered response suggestions and emotional state analysis</p>
      </header>

      <div className="app-content">
        <div className="left-panel">
          <EmotionalStateDisplay emotionalState={emotionalState} />
          <AudioRecorder 
            onTranscription={handleAudioTranscription}
          />
        </div>

        <div className="center-panel">
          <ConversationDisplay conversation={conversation} />
          <KeywordInput 
            value={userKeywords}
            onChange={setUserKeywords}
            onGenerate={handleGenerateResponses}
            isProcessing={isProcessing}
          />
          {generatedResponses && (
            <ResponseSelector 
              responses={generatedResponses}
              onSelect={handleResponseSelect}
            />
          )}
        </div>
      </div>
    </div>
  )
}

export default App
