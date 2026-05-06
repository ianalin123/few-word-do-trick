import React, { useState, useEffect } from 'react'
import AudioRecorder from './components/AudioRecorder'
import ConversationDisplay from './components/ConversationDisplay'
import ResponseSelector from './components/ResponseSelector'
import EmotionalStateDisplay from './components/EmotionalStateDisplay'
import KeywordInput from './components/KeywordInput'
import FeedbackModal from './components/FeedbackModal'
import './App.css'

// Get or create a unique user ID
function getUserId() {
  let userId = localStorage.getItem('user_id')
  if (!userId) {
    userId = 'user_' + Math.random().toString(36).substring(2, 15)
    localStorage.setItem('user_id', userId)
  }
  return userId
}

function App() {
  const [conversation, setConversation] = useState([])
  const [emotionalState, setEmotionalState] = useState('neutral')
  const [userKeywords, setUserKeywords] = useState('')
  const [generatedResponses, setGeneratedResponses] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [showFeedbackModal, setShowFeedbackModal] = useState(false)

  // Add personality state
  const [personalityType, setPersonalityType] = useState('')
  const [personalityDescription, setPersonalityDescription] = useState('')

  const [editableResponse, setEditableResponse] = useState('')
  const [selectedEnergy, setSelectedEnergy] = useState('')
  const [uiStep, setUiStep] = useState('input') // 'input' | 'selecting' | 'editing'

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
          emotional_state: emotionalState,
          personality_type: personalityType, // Send personality type
          personality_description: personalityDescription // Send description
        })
      })

      if (!response.ok) {
        throw new Error('Failed to generate responses')
      }

      const data = await response.json()
      setGeneratedResponses(data)
      setUiStep('selecting')
    } catch (error) {
      console.error('Error generating responses:', error)
      alert('Failed to generate responses. Please try again.')
    } finally {
      setIsProcessing(false)
    }
  }

  // Add function to handle personality results from quiz
  const handlePersonalityResult = (type, description) => {
    setPersonalityType(type)
    setPersonalityDescription(description)
  }

  const handleResponseSelect = (response, energy) => {
    setEditableResponse(response)
    setSelectedEnergy(energy)
    setUiStep('editing')
  }

  const handleBackToInput = () => {
    setGeneratedResponses(null)
    setUserKeywords('')
    setUiStep('input')
  }

  const handleBackToSelecting = () => {
    setEditableResponse('')
    setSelectedEnergy('')
    setUiStep('selecting')
  }

  // Handle sending the edited response
  const handleSendResponse = async () => {
    if (!editableResponse.trim()) {
      return
    }

    // Add user's edited response to conversation
    const userMessage = {
      id: Date.now(),
      speaker: 'USER',
      text: editableResponse,
      timestamp: new Date().toLocaleTimeString()
    }
    setConversation(prev => [...prev, userMessage])

    // Play text-to-speech with energy level and emotional state
    try {
      const userId = getUserId()
      const formData = new FormData()
      formData.append('text', editableResponse)
      formData.append('energy', selectedEnergy)
      formData.append('emotional_state', emotionalState)
      formData.append('user_id', userId)

      console.log(`🎤 TTS: energy=${selectedEnergy}, emotion=${emotionalState}, user=${userId}`)

      const response_audio = await fetch('/api/text-to-speech', {
        method: 'POST',
        body: formData
      })

      if (response_audio.ok) {
        const audioBlob = await response_audio.blob()
        const audioUrl = URL.createObjectURL(audioBlob)
        const audio = new Audio(audioUrl)
        audio.play()

        // Clean up URL after audio finishes
        audio.onended = () => URL.revokeObjectURL(audioUrl)
      } else {
        console.error('TTS failed:', await response_audio.text())
      }
    } catch (error) {
      console.error('Error playing audio:', error)
    }

    // Clear everything and return to input
    setGeneratedResponses(null)
    setUserKeywords('')
    setEditableResponse('')
    setSelectedEnergy('')
    setUiStep('input')
  }

  const clearConversation = () => {
    setConversation([])
    setGeneratedResponses(null)
    setUserKeywords('')
    setEditableResponse('')
    setSelectedEnergy('')
    setUiStep('input')
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Why waste time say lot word when few word do trick?</h1>
        <p>Real-time conversation assistant with AI-powered response suggestions and emotional state analysis</p>
        
        {/* Show personality info if available */}
        {personalityType && (
          <div className="personality-display">
            <span>🧠 Your Type: {personalityType} - {personalityDescription}</span>
          </div>
        )}
        
        <div className="header-actions">
          <button 
            className="action-button clear-btn"
            onClick={clearConversation}
            title="Clear conversation and start fresh"
          >
            🗑️ Clear All
          </button>
          <button 
            className="action-button feedback-btn"
            onClick={() => setShowFeedbackModal(true)}
            title="Take a personality quiz"
          >
            🧠 Personality Quiz
          </button>
        </div>
      </header>

      <div className="voice-dashboard-container">
        <button
          onClick={() => window.open('/voices', '_blank')}
          className="voice-dashboard-button"
        >
          🎙️ Voice Dashboard
        </button>
      </div>


      <div className="app-content">
        <div className="left-panel">
          <EmotionalStateDisplay emotionalState={emotionalState} />
          <AudioRecorder 
            onTranscription={handleAudioTranscription}
          />
        </div>

        <div className="center-panel">
          {uiStep === 'input' && (
            <>
              <ConversationDisplay conversation={conversation} />
              <KeywordInput
                value={userKeywords}
                onChange={setUserKeywords}
                onGenerate={handleGenerateResponses}
                isProcessing={isProcessing}
              />
            </>
          )}
          {uiStep === 'selecting' && (
            <ResponseSelector
              responses={generatedResponses}
              onSelect={handleResponseSelect}
              onBack={handleBackToInput}
              emotionalState={emotionalState}
            />
          )}
          {uiStep === 'editing' && (
            <div className="response-editor">
              <button className="back-button-editor" onClick={handleBackToSelecting}>← Back</button>
              <h3>Edit Your Response</h3>
              <textarea
                value={editableResponse}
                onChange={(e) => setEditableResponse(e.target.value)}
                className="response-textarea"
                placeholder="Edit your response before sending..."
                rows={4}
              />
              <div className="editor-actions">
                <button
                  onClick={handleSendResponse}
                  className="send-response-button"
                  disabled={!editableResponse.trim()}
                >
                  🎤 Send & Speak
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Updated Feedback Modal */}
      {showFeedbackModal && (
        <FeedbackModal 
          onClose={() => setShowFeedbackModal(false)}
          onPersonalityResult={handlePersonalityResult}
        />
      )}
    </div>
  )
}

export default App