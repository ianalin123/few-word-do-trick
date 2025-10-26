import React from 'react'
import './ResponseSelector.css'

const ResponseSelector = ({ responses, onSelect, emotionalState }) => {
  const handleSelect = (energy) => {
    const response = responses[energy]
    if (response) {
      onSelect(response, energy)  // Pass energy level (low/medium/high/contradictory)
    }
  }

  // Get emoji based on emotional state
  const getEmoji = (energy) => {
    const emojiMap = {
      low: emotionalState === 'happy' ? '😌' : '😔',
      medium: emotionalState === 'happy' ? '😊' : '😟',
      high: emotionalState === 'happy' ? '🤩' : '😢',
      contradictory: '😏'  // Sarcasm is always the same emoji
    }
    return emojiMap[energy] || '😐'
  }

  return (
    <div className="response-selector">
      <h3>Choose Your Response</h3>
      <div className="emotional-context">
        Current emotion: <strong>{emotionalState}</strong>
      </div>
      <div className="response-options">
        <div 
          className="response-option low" 
          onClick={() => handleSelect('low')}
        >
          <div className="response-label">
            {getEmoji('low')} Low Energy
          </div>
          <div className="response-text">{responses.low}</div>
        </div>
        
        <div 
          className="response-option medium" 
          onClick={() => handleSelect('medium')}
        >
          <div className="response-label">
            {getEmoji('medium')} Medium Energy
          </div>
          <div className="response-text">{responses.medium}</div>
        </div>
        
        <div 
          className="response-option high" 
          onClick={() => handleSelect('high')}
        >
          <div className="response-label">
            {getEmoji('high')} High Energy
          </div>
          <div className="response-text">{responses.high}</div>
        </div>
        
        <div 
          className="response-option contradictory" 
          onClick={() => handleSelect('contradictory')}
        >
          <div className="response-label">
            {getEmoji('contradictory')} Contradictory
          </div>
          <div className="response-text">{responses.contradictory}</div>
        </div>
      </div>
    </div>
  )
}

export default ResponseSelector