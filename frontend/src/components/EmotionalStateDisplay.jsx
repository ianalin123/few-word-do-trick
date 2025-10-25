import React from 'react'

const EmotionalStateDisplay = ({ emotionalState }) => {
  const getKevinImage = (state) => {
    const imageMap = {
      happy: '/kevin-happy.png',
      neutral: '/kevin-neutral.png',
      sad: '/kevin-sad.png'
    }
    return imageMap[state] || '/kevin-neutral.png'
  }

  const getDescription = (state) => {
    const descriptions = {
      happy: 'Kevin is feeling great!',
      neutral: 'Kevin is just chillin\'',
      sad: 'Kevin is having a rough day'
    }
    return descriptions[state] || 'Kevin is... Kevin'
  }

  const getKevinQuote = (state) => {
    const quotes = {
      happy: '"I have very little patience for stupidity."',
      neutral: '"I work hard, I play hard."',
      sad: '"I just want to sit on the beach and eat hot dogs."'
    }
    return quotes[state] || '"I am Kevin Malone."'
  }

  return (
    <div className={`emotional-state ${emotionalState}`}>
      <h3>Kevin's Current Mood</h3>
      <div className="kevin-image-container">
        <img 
          src={getKevinImage(emotionalState)} 
          alt={`Kevin ${emotionalState}`}
          className="kevin-image"
        />
      </div>
      <div className="emotional-label">
        {emotionalState.charAt(0).toUpperCase() + emotionalState.slice(1)} Kevin
      </div>
      <div className="emotional-description">
        {getDescription(emotionalState)}
      </div>
      <div className="kevin-quote">
        {getKevinQuote(emotionalState)}
      </div>
    </div>
  )
}

export default EmotionalStateDisplay
