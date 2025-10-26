import React, { useState, useEffect } from 'react'
import AudioRecorder from './components/AudioRecorder'
import ConversationDisplay from './components/ConversationDisplay'
import ResponseSelector from './components/ResponseSelector'
import EmotionalStateDisplay from './components/EmotionalStateDisplay'
import KeywordInput from './components/KeywordInput'
import FeedbackModal from './components/FeedbackModal'
import './App.css'

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

  // Simulate EEG data - in real implementation, this would come from EEG device
  useEffect(() => {
    const interval = setInterval(() => {
      const states = ['happy', 'neutral', 'sad']
      const randomState = states[Math.floor(Math.random() * states.length)]
      setEmotionalState(randomState)
    }, 5000) // Update every 5 seconds

    return () => clearInterval(interval)
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

  const clearConversation = () => {
    setConversation([])
    setGeneratedResponses(null)
    setUserKeywords('')
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