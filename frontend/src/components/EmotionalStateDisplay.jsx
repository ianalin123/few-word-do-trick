import React from 'react'

const EmotionalStateDisplay = ({ emotionalState }) => {
  const getEmoji = (state) => {
    const emojiMap = {
      happy: '😊',
      excited: '🤩',
      calm: '😌',
      neutral: '😐',
      focused: '🧠',
      sad: '😢',
      angry: '😠',
      surprised: '😲'
    }
    return emojiMap[state] || '😐'
  }

  const getDescription = (state) => {
    const descriptions = {
      happy: 'Feeling positive and cheerful',
      excited: 'Energized and enthusiastic',
      calm: 'Relaxed and peaceful',
      neutral: 'Balanced and steady',
      focused: 'Concentrated and alert',
      sad: 'Feeling down or melancholy',
      angry: 'Frustrated or upset',
      surprised: 'Caught off guard'
    }
    return descriptions[state] || 'Unknown emotional state'
  }

  return (
    <div className={`emotional-state ${emotionalState}`}>
      <h3>Current Emotional State</h3>
      <div className="emotional-indicator">
        {getEmoji(emotionalState)}
      </div>
      <div className="emotional-label">
        {emotionalState.charAt(0).toUpperCase() + emotionalState.slice(1)}
      </div>
      <div className="emotional-description">
        {getDescription(emotionalState)}
      </div>
    </div>
  )
}

export default EmotionalStateDisplay
