import React, { useState } from 'react'
import './FeedbackModal.css'

function FeedbackModal({ onClose, onPersonalityResult }) {
  const [answers, setAnswers] = useState({
    q1: '',
    q2: '',
    q3: '',
    q4: '',
    q5: ''
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [personalityResult, setPersonalityResult] = useState('')

  const questions = [
    {
      id: 'q1',
      question: 'How do you usually recharge after a busy week?',
      options: [
        { value: 'E', text: 'Hanging out with friends, talking, or doing something social' },
        { value: 'I', text: 'Spending time alone to relax and reflect' }
      ]
    },
    {
      id: 'q2',
      question: 'When learning something new, what do you focus on first?',
      options: [
        { value: 'S', text: 'The practical facts, real examples, and how it works right now' },
        { value: 'N', text: 'The deeper meaning, possibilities, and what it could lead to' }
      ]
    },
    {
      id: 'q3',
      question: 'How do you prefer to make decisions?',
      options: [
        { value: 'T', text: 'By using logic and objective reasoning' },
        { value: 'F', text: 'By considering people\'s feelings and personal values' }
      ]
    },
    {
      id: 'q4',
      question: 'Which best describes your work or study style?',
      options: [
        { value: 'J', text: 'I like to plan ahead, make lists, and stick to schedules' },
        { value: 'P', text: 'I prefer flexibility and adjusting plans as I go' }
      ]
    },
    {
      id: 'q5',
      question: 'In a group project, what role do you naturally take?',
      options: [
        { value: 'J', text: 'The organizer — I set goals, make decisions, and keep everyone on track' },
        { value: 'P', text: 'The adapter — I go with the flow, offer ideas, and help where needed' }
      ]
    }
  ]

  const handleAnswerChange = (questionId, value) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: value
    }))
  }

  const calculatePersonality = () => {
    const traits = {
      E: 0, I: 0,
      S: 0, N: 0,
      T: 0, F: 0,
      J: 0, P: 0
    }

    // Count answers
    Object.values(answers).forEach(answer => {
      if (traits.hasOwnProperty(answer)) {
        traits[answer]++
      }
    })

    // Determine personality type
    const type = 
      (traits.E >= traits.I ? 'E' : 'I') +
      (traits.S >= traits.N ? 'S' : 'N') +
      (traits.T >= traits.F ? 'T' : 'F') +
      (traits.J >= traits.P ? 'J' : 'P')

    return type
  }

  const getPersonalityDescription = (type) => {
    const descriptions = {
      'ESTJ': 'The Executive - Organized, practical, and results-oriented',
      'ESTP': 'The Entrepreneur - Energetic, flexible, and action-oriented',
      'ESFJ': 'The Consul - Caring, social, and harmony-seeking',
      'ESFP': 'The Entertainer - Enthusiastic, creative, and people-focused',
      'ENTJ': 'The Commander - Strategic, decisive, and natural leader',
      'ENTP': 'The Debater - Innovative, curious, and idea-generating',
      'ENFJ': 'The Protagonist - Inspiring, empathetic, and people-developer',
      'ENFP': 'The Campaigner - Enthusiastic, creative, and possibility-focused',
      'ISTJ': 'The Logistician - Responsible, detailed, and tradition-minded',
      'ISTP': 'The Virtuoso - Practical, hands-on, and adaptable',
      'ISFJ': 'The Protector - Caring, detail-oriented, and supportive',
      'ISFP': 'The Adventurer - Gentle, flexible, and value-driven',
      'INTJ': 'The Architect - Strategic, independent, and visionary',
      'INTP': 'The Thinker - Analytical, curious, and theory-focused',
      'INFJ': 'The Advocate - Insightful, principled, and future-focused',
      'INFP': 'The Mediator - Idealistic, creative, and value-driven'
    }
    
    return descriptions[type] || 'Unique personality type!'
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    // Check if all questions are answered
    const unanswered = questions.find(q => !answers[q.id])
    if (unanswered) {
      alert('Please answer all questions to see your personality type!')
      return
    }

    setIsSubmitting(true)
    
    try {
      // Calculate personality type
      const type = calculatePersonality()
      const description = getPersonalityDescription(type)
      setPersonalityResult(`${type}: ${description}`)
      
      // Pass results back to parent component
      onPersonalityResult(type, description)
      
      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      setShowResults(true)
    } catch (error) {
      alert('Something went wrong. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const resetQuiz = () => {
    setAnswers({
      q1: '',
      q2: '',
      q3: '',
      q4: '',
      q5: ''
    })
    setShowResults(false)
    setPersonalityResult('')
  }

  if (showResults) {
    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="modal-content" onClick={e => e.stopPropagation()}>
          <div className="modal-header">
            <h2>🎉 Your Personality Type</h2>
            <button className="close-button" onClick={onClose}>×</button>
          </div>

          <div className="results-content">
            <div className="personality-result">
              <h3>{personalityResult}</h3>
              <p>This is based on a simplified version of the Myers-Briggs Type Indicator (MBTI).</p>
              
              <div className="result-actions">
                <button className="reset-button" onClick={resetQuiz}>
                  🔄 Take Quiz Again
                </button>
                <button className="close-result-button" onClick={onClose}>
                  ✨ Continue to App
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>🧠 Personality Quiz</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        <form onSubmit={handleSubmit} className="personality-form">
          <div className="quiz-intro">
            <p>Discover your personality type! Answer these 5 questions to learn more about yourself.</p>
          </div>

          {questions.map((question, index) => (
            <div key={question.id} className="question-group">
              <h4>Q{index + 1}. {question.question}</h4>
              <div className="options-group">
                {question.options.map((option, optionIndex) => (
                  <label key={optionIndex} className="option-label">
                    <input
                      type="radio"
                      name={question.id}
                      value={option.value}
                      checked={answers[question.id] === option.value}
                      onChange={(e) => handleAnswerChange(question.id, e.target.value)}
                    />
                    <span className="option-text">
                      <strong>{String.fromCharCode(65 + optionIndex)}.</strong> {option.text}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          ))}

          <div className="form-actions">
            <button 
              type="button" 
              className="cancel-button"
              onClick={onClose}
            >
              Cancel
            </button>
            <button 
              type="submit" 
              className="submit-button"
              disabled={isSubmitting}
            >
              {isSubmitting ? '🧠 Analyzing...' : '🎯 Get My Personality Type'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default FeedbackModal