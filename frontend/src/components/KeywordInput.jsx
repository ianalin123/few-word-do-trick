import React from 'react'

const KeywordInput = ({ value, onChange, onGenerate, isProcessing }) => {
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !isProcessing) {
      onGenerate()
    }
  }

  return (
    <div className="keyword-input">
      <h3>Keywords & Response Generation</h3>
      <div className="input-group">
        <input
          type="text"
          className="keyword-field"
          placeholder="Enter a few key words to generate responses..."
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isProcessing}
        />
        <button
          className="generate-button"
          onClick={onGenerate}
          disabled={isProcessing || !value.trim()}
        >
          {isProcessing ? 'Generating...' : 'Generate'}
        </button>
      </div>
      <p style={{ fontSize: '0.9rem', opacity: 0.7, margin: 0 }}>
        Enter keywords to generate AI responses based on your conversation and emotional state.
      </p>
    </div>
  )
}

export default KeywordInput
