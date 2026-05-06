import React, { useEffect, useState } from 'react'
import axios from 'axios'
import './VoiceDashboard.css'

// Get or create a unique user ID
function getUserId() {
  let userId = localStorage.getItem('user_id')
  if (!userId) {
    userId = 'user_' + Math.random().toString(36).substring(2, 15)
    localStorage.setItem('user_id', userId)
  }
  return userId
}

function VoiceDashboard() {
  const [voices, setVoices] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [playingVoiceId, setPlayingVoiceId] = useState(null)
  const [selectedVoiceId, setSelectedVoiceId] = useState(null)
  const [selectedVoiceName, setSelectedVoiceName] = useState(null)
  const [testingSettings, setTestingSettings] = useState(false)

  // Voice settings state
  const [settings, setSettings] = useState({
    stability: 0.5,
    similarity_boost: 0.75,
    style: 0.0,
    use_speaker_boost: true
  })
  const [savingSettings, setSavingSettings] = useState(false)
  const [showTooltip, setShowTooltip] = useState(null)

  useEffect(() => {
    async function fetchVoices() {
      try {
        console.log('🔑 Using API Key:', import.meta.env.VITE_ELEVENLABS_API_KEY)
        const res = await axios.get('https://api.elevenlabs.io/v1/voices', {
          headers: {
            'xi-api-key': import.meta.env.VITE_ELEVENLABS_API_KEY,
          },
        })
        console.log('✅ Voices response:', res.data)
        setVoices((res.data.voices || []).filter(v => v.category === 'premade'))
      } catch (err) {
        console.error('❌ Error fetching voices:', err)
        setError('Failed to load voices. Check your API key or network.')
      } finally {
        setLoading(false)
      }
    }

    async function fetchCurrentVoice() {
      try {
        const userId = getUserId()
        const res = await axios.get(`/api/user/${userId}/voice`)
        if (res.data.voice_id) {
          setSelectedVoiceId(res.data.voice_id)
          setSelectedVoiceName(res.data.voice_name)
          console.log('✅ Current voice:', res.data.voice_name)
        }
      } catch (err) {
        console.error('❌ Error fetching current voice:', err)
      }
    }

    async function fetchVoiceSettings() {
      try {
        const userId = getUserId()
        const res = await axios.get(`/api/user/${userId}/voice-settings`)
        setSettings(res.data)
        console.log('✅ Voice settings loaded:', res.data)
      } catch (err) {
        console.error('❌ Error fetching voice settings:', err)
      }
    }

    fetchVoices()
    fetchCurrentVoice()
    fetchVoiceSettings()
  }, [])

  const generatePreviewWithTTS = async (voice) => {
    try {
      console.log('🎧 Generating preview using TTS API for:', voice.name)
      const response = await axios.post(
        `https://api.elevenlabs.io/v1/text-to-speech/${voice.voice_id}`,
        {
          text: "Hello! This is a preview of my voice.",
          model_id: "eleven_monolingual_v1",
        },
        {
          headers: {
            'xi-api-key': import.meta.env.VITE_ELEVENLABS_API_KEY,
            'Content-Type': 'application/json',
          },
          responseType: 'blob',
        }
      )

      const audioUrl = URL.createObjectURL(response.data)
      const audio = new Audio(audioUrl)
      audio.onended = () => {
        setPlayingVoiceId(null)
        URL.revokeObjectURL(audioUrl)
      }
      audio.onerror = () => {
        setPlayingVoiceId(null)
        URL.revokeObjectURL(audioUrl)
      }
      await audio.play()
    } catch (error) {
      console.error('🎧 TTS Error:', error)
      setPlayingVoiceId(null)
      if (error.response?.status === 401) {
        alert('❌ API Key invalid or expired')
      } else if (error.response?.status === 429) {
        alert('❌ API quota exceeded, please try again later')
      } else {
        alert('❌ Failed to generate preview: ' + (error.response?.data?.detail?.message || error.message))
      }
    }
  }

  const handlePreview = async (voice) => {
    try {
      setPlayingVoiceId(voice.voice_id)

      if (voice.preview_url) {
        console.log('🎧 Using preview_url:', voice.preview_url)

        const audio = new Audio(voice.preview_url)
        audio.onended = () => {
          setPlayingVoiceId(null)
        }
        audio.onerror = (e) => {
          console.error('❌ Preview URL error, trying TTS API...', e)
          generatePreviewWithTTS(voice)
        }
        await audio.play()
      } else {
        await generatePreviewWithTTS(voice)
      }
    } catch (error) {
      console.error('🎧 Error playing preview:', error)
      setPlayingVoiceId(null)
      alert('❌ Playback failed, please try again')
    }
  }

  const handleSaveSettings = async () => {
    setSavingSettings(true)
    try {
      const userId = getUserId()
      await axios.post('/api/user/voice-settings', {
        user_id: userId,
        ...settings
      })
      alert('✅ Voice settings saved!')
    } catch (error) {
      console.error('Error saving settings:', error)
      alert('❌ Failed to save settings')
    } finally {
      setSavingSettings(false)
    }
  }

  const handleResetSettings = () => {
    setSettings({
      stability: 0.5,
      similarity_boost: 0.75,
      style: 0.0,
      use_speaker_boost: true
    })
  }

  const handleTestSettings = async () => {
    if (!selectedVoiceId) {
      alert('Select a voice first')
      return
    }
    setTestingSettings(true)
    try {
      const response = await axios.post(
        `https://api.elevenlabs.io/v1/text-to-speech/${selectedVoiceId}`,
        {
          text: "Hey, this is how I sound with your current settings. How does this feel?",
          model_id: "eleven_turbo_v2_5",
          voice_settings: settings
        },
        {
          headers: {
            'xi-api-key': import.meta.env.VITE_ELEVENLABS_API_KEY,
            'Content-Type': 'application/json',
          },
          responseType: 'blob'
        }
      )
      const audioUrl = URL.createObjectURL(response.data)
      const audio = new Audio(audioUrl)
      audio.onended = () => URL.revokeObjectURL(audioUrl)
      await audio.play()
    } catch (err) {
      console.error('Test failed:', err)
      alert('Test failed — check console')
    } finally {
      setTestingSettings(false)
    }
  }

  const tooltips = {
    stability: {
      title: "Stability",
      description: "Controls consistency vs. expressiveness. Lower = more varied/emotional, Higher = more consistent/predictable. Ranges: Low (0-0.33), Medium (0.34-0.66), High (0.67-1.0)"
    },
    similarity_boost: {
      title: "Similarity Boost",
      description: "How closely the voice matches the original characteristics. Higher values maintain voice identity better. Recommended: 0.75-0.85"
    },
    style: {
      title: "Style Exaggeration",
      description: "Amplifies the speaking style and personality. 0 = neutral/natural. Higher values add more character but increase processing time. Ranges apply per energy level."
    },
    use_speaker_boost: {
      title: "Speaker Boost",
      description: "Enhances voice clarity and quality. Recommended to keep enabled for best results."
    }
  }

  if (loading) return <div className="loading-state">Loading voices...</div>
  if (error) return <div className="error-state">{error}</div>

  return (
    <div className="voice-dashboard">
      <div className="voice-dashboard-header">
        <button
          onClick={() => window.location.href = '/'}
          className="back-button"
        >
          ← Back to Home
        </button>
      </div>

      <h1>🎙️ Voice Dashboard</h1>
      <p className="voice-dashboard-subtitle">
        {selectedVoiceName
          ? <>Active voice: <span className="selected-voice-name">{selectedVoiceName}</span></>
          : 'Select a voice and customize its settings below.'}
      </p>

      {/* Voice Settings Section */}
      <div className="settings-container">
        <h2>🎛️ Voice Settings</h2>
        <p className="settings-description">
          Customize how your voice sounds. These base settings will be automatically adjusted based on energy levels.
        </p>

        <div className="energy-explanation">
          <h3>📊 Energy Ranges Explanation</h3>
          <ul>
            <li><strong>Low Energy (0-0.33):</strong> Calm, subdued delivery</li>
            <li><strong>Medium Energy (0.34-0.66):</strong> Balanced, natural speech</li>
            <li><strong>High Energy (0.67-1.0):</strong> Expressive, dynamic delivery</li>
            <li><strong>Contradictory (0.1):</strong> Flat, sarcastic tone (fixed at 0.1)</li>
          </ul>
        </div>

        {/* Stability Slider */}
        <div className="slider-control">
          <div className="slider-header">
            <label className="slider-label">{tooltips.stability.title}</label>
            <button
              onMouseEnter={() => setShowTooltip('stability')}
              onMouseLeave={() => setShowTooltip(null)}
              className="tooltip-button"
            >
              ?
            </button>
          </div>
          {showTooltip === 'stability' && (
            <div className="tooltip-content">
              {tooltips.stability.description}
            </div>
          )}
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={settings.stability}
            onChange={(e) => setSettings({ ...settings, stability: parseFloat(e.target.value) })}
            className="slider-input"
          />
          <div className="slider-labels">
            <span>0.0 (Very Expressive)</span>
            <span className="slider-value">{settings.stability.toFixed(2)}</span>
            <span>1.0 (Very Consistent)</span>
          </div>
        </div>

        {/* Similarity Boost Slider */}
        <div className="slider-control">
          <div className="slider-header">
            <label className="slider-label">{tooltips.similarity_boost.title}</label>
            <button
              onMouseEnter={() => setShowTooltip('similarity_boost')}
              onMouseLeave={() => setShowTooltip(null)}
              className="tooltip-button"
            >
              ?
            </button>
          </div>
          {showTooltip === 'similarity_boost' && (
            <div className="tooltip-content">
              {tooltips.similarity_boost.description}
            </div>
          )}
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={settings.similarity_boost}
            onChange={(e) => setSettings({ ...settings, similarity_boost: parseFloat(e.target.value) })}
            className="slider-input"
          />
          <div className="slider-labels">
            <span>0.0 (Less Similar)</span>
            <span className="slider-value">{settings.similarity_boost.toFixed(2)}</span>
            <span>1.0 (More Similar)</span>
          </div>
        </div>

        {/* Style Slider */}
        <div className="slider-control">
          <div className="slider-header">
            <label className="slider-label">{tooltips.style.title}</label>
            <button
              onMouseEnter={() => setShowTooltip('style')}
              onMouseLeave={() => setShowTooltip(null)}
              className="tooltip-button"
            >
              ?
            </button>
          </div>
          {showTooltip === 'style' && (
            <div className="tooltip-content">
              {tooltips.style.description}
            </div>
          )}
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={settings.style}
            onChange={(e) => setSettings({ ...settings, style: parseFloat(e.target.value) })}
            className="slider-input"
          />
          <div className="slider-labels">
            <span>0.0 (Neutral/Fast)</span>
            <span className="slider-value">{settings.style.toFixed(2)}</span>
            <span>1.0 (Exaggerated/Slow)</span>
          </div>
        </div>

        {/* Speaker Boost Toggle */}
        <div className="checkbox-control">
          <div className="slider-header">
            <label className="slider-label">{tooltips.use_speaker_boost.title}</label>
            <button
              onMouseEnter={() => setShowTooltip('use_speaker_boost')}
              onMouseLeave={() => setShowTooltip(null)}
              className="tooltip-button"
            >
              ?
            </button>
          </div>
          {showTooltip === 'use_speaker_boost' && (
            <div className="tooltip-content">
              {tooltips.use_speaker_boost.description}
            </div>
          )}
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={settings.use_speaker_boost}
              onChange={(e) => setSettings({ ...settings, use_speaker_boost: e.target.checked })}
              className="checkbox-input"
            />
            <span className="checkbox-text">{settings.use_speaker_boost ? 'Enabled (Recommended)' : 'Disabled'}</span>
          </label>
        </div>

        {/* Action Buttons */}
        <div className="settings-actions">
          <button
            onClick={handleTestSettings}
            disabled={testingSettings || !selectedVoiceId}
            className="test-button"
          >
            {testingSettings ? '🔊 Playing...' : '🎧 Test Current Settings'}
          </button>
          <button
            onClick={handleSaveSettings}
            disabled={savingSettings}
            className="save-button"
          >
            {savingSettings ? '💾 Saving...' : '💾 Save Settings'}
          </button>
          <button
            onClick={handleResetSettings}
            className="reset-button"
          >
            🔄 Reset
          </button>
        </div>
      </div>

      {/* Voice Selection Section */}
      <div className="voices-section">
        <h2>🎤 Available Voices</h2>
        <p className="voices-description">Preview and select your preferred voice.</p>

        <div className="voices-grid">
          {voices.map((v) => (
            <div
              key={v.voice_id}
              className={`voice-card ${playingVoiceId === v.voice_id ? 'playing' : ''} ${selectedVoiceId === v.voice_id ? 'selected' : ''}`}
            >
              <h3>{v.name}</h3>
              <p className="voice-category">
                Category: {v.category || 'Unknown'}
              </p>
              {v.labels && (
                <p className="voice-labels">
                  {v.labels.accent && `${v.labels.accent}`}
                  {v.labels.age && ` • ${v.labels.age}`}
                  {v.labels.gender && ` • ${v.labels.gender}`}
                </p>
              )}
              <div className="voice-actions">
                <button
                  onClick={() => handlePreview(v)}
                  disabled={playingVoiceId === v.voice_id}
                  className="preview-button"
                >
                  {playingVoiceId === v.voice_id ? '🔊 Playing...' : '▶️ Preview'}
                </button>
                <button
                  onClick={async () => {
                    try {
                      const userId = getUserId()
                      await axios.post('/api/user/voice', {
                        user_id: userId,
                        voice_id: v.voice_id,
                        voice_name: v.name
                      })
                      setSelectedVoiceId(v.voice_id)
                      setSelectedVoiceName(v.name)
                      localStorage.setItem('selectedVoice', v.voice_id)
                      localStorage.setItem('selectedVoiceName', v.name)

                      // Apply this voice's preset settings if available, else reset to defaults
                      const newSettings = v.settings ? {
                        stability: v.settings.stability ?? 0.5,
                        similarity_boost: v.settings.similarity_boost ?? 0.75,
                        style: v.settings.style ?? 0.0,
                        use_speaker_boost: v.settings.use_speaker_boost ?? true
                      } : {
                        stability: 0.5,
                        similarity_boost: 0.75,
                        style: 0.0,
                        use_speaker_boost: true
                      }
                      setSettings(newSettings)
                      await axios.post('/api/user/voice-settings', { user_id: userId, ...newSettings })

                      alert(`✅ Selected voice: ${v.name}`)
                    } catch (error) {
                      console.error('❌ Error saving voice selection:', error)
                      alert('❌ Failed to save voice selection')
                    }
                  }}
                  className={`select-button ${selectedVoiceId === v.voice_id ? 'selected' : ''}`}
                >
                  {selectedVoiceId === v.voice_id ? '✓ Selected' : '✅ Select'}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default VoiceDashboard
