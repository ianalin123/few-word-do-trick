import React, { useState } from 'react'
import './ResponseSelector.css'

const ResponseSelector = ({ responses, onSelect, onBack, emotionalState }) => {
  const [highlightedOption, setHighlightedOption] = useState(null)

  const handleSelect = (energy) => {
    const response = responses[energy]
    if (!response) return
    setHighlightedOption(energy)
    setTimeout(() => onSelect(response, energy), 150)
  }

  const getEmoji = (energy) => {
    const emojiMap = {
      low: emotionalState === 'happy' ? '😌' : '😔',
      medium: emotionalState === 'happy' ? '😊' : '😟',
      high: emotionalState === 'happy' ? '🤩' : '😢',
      contradictory: '😏'
    }
    return emojiMap[energy] || '😐'
  }

  return (
    <div className="response-selector">
      <div className="response-selector-header">
        <button className="back-button-selector" onClick={onBack}>← Back</button>
        <h3>Select a Response to Edit</h3>
      </div>
      <div className="response-options">
        {['low', 'medium', 'high', 'contradictory'].map((energy) => (
          <div
            key={energy}
            className={`response-option ${energy} ${highlightedOption === energy ? 'selected' : ''}`}
            onClick={() => handleSelect(energy)}
          >
            <div className="response-label">
              {getEmoji(energy)} {energy.charAt(0).toUpperCase() + energy.slice(1)} {energy !== 'contradictory' ? 'Energy' : ''}
            </div>
            <div className="response-text">{responses[energy]}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ResponseSelector
