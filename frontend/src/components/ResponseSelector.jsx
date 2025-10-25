import React from 'react'

const ResponseSelector = ({ responses, onSelect }) => {
  const handleSelect = (sentiment) => {
    const response = responses[sentiment]
    onSelect(response, sentiment)
  }

  return (
    <div className="response-selector">
      <h3>Choose Your Response</h3>
      <div className="response-options">
        <div 
          className="response-option calm"
          onClick={() => handleSelect('calm')}
        >
          <div className="response-label">Calm</div>
          <div className="response-text">{responses.calm}</div>
        </div>
        
        <div 
          className="response-option neutral"
          onClick={() => handleSelect('neutral')}
        >
          <div className="response-label">Neutral</div>
          <div className="response-text">{responses.neutral}</div>
        </div>
        
        <div 
          className="response-option excited"
          onClick={() => handleSelect('excited')}
        >
          <div className="response-label">Excited</div>
          <div className="response-text">{responses.excited}</div>
        </div>
      </div>
    </div>
  )
}

export default ResponseSelector
