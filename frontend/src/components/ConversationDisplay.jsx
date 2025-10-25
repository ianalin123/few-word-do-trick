import React from 'react'

const ConversationDisplay = ({ conversation }) => {
  return (
    <div className="conversation">
      <h3>Conversation</h3>
      {conversation.length === 0 ? (
        <p style={{ opacity: 0.6, fontStyle: 'italic' }}>
          Start a conversation by recording audio or typing keywords...
        </p>
      ) : (
        <div className="conversation-messages">
          {conversation.map((message) => (
            <div key={message.id} className={`message ${message.speaker}`}>
              <div className="message-header">
                <span className="speaker">{message.speaker}</span>
                <span className="timestamp">{message.timestamp}</span>
              </div>
              <div className="message-text">{message.text}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ConversationDisplay
