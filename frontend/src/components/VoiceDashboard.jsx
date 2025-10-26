import React, { useEffect, useState } from 'react'
import axios from 'axios'

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
        setVoices(res.data.voices || [])
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
        alert('❌ API Key 无效或已过期')
      } else if (error.response?.status === 429) {
        alert('❌ API 配额已用完，请稍后再试')
      } else {
        alert('❌ 生成预览失败：' + (error.response?.data?.detail?.message || error.message))
      }
    }
  }

  const handlePreview = async (voice) => {
    try {
      setPlayingVoiceId(voice.voice_id)

      if (voice.preview_url) {
        console.log('🎧 Using preview_url:', voice.preview_url)

        // ✅ 直接用 Audio 播放 preview_url（不经过 axios，避免 CORS）
        const audio = new Audio(voice.preview_url)
        audio.onended = () => {
          setPlayingVoiceId(null)
        }
        audio.onerror = (e) => {
          console.error('❌ Preview URL error, trying TTS API...', e)
          // 如果 preview_url 失败，尝试用 TTS API
          generatePreviewWithTTS(voice)
        }
        await audio.play()
      } else {
        // 没有 preview_url，用 TTS API
        await generatePreviewWithTTS(voice)
      }
    } catch (error) {
      console.error('🎧 Error playing preview:', error)
      setPlayingVoiceId(null)
      alert('❌ 播放失败，请重试')
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

  if (loading) return <p>Loading voices...</p>
  if (error) return <p style={{ color: 'red' }}>{error}</p>

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
      <div style={{ marginBottom: '1.5rem' }}>
        <button
          onClick={() => window.location.href = '/'}
          style={{
            padding: '10px 20px',
            borderRadius: '6px',
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            cursor: 'pointer',
            fontSize: '14px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
        >
          ← Back to Home
        </button>
      </div>

      <h1>🎙️ Voice Dashboard</h1>
      <p style={{ marginBottom: '2rem' }}>Select a voice and customize its settings below.</p>

      {/* Voice Settings Section */}
      <div style={{
        backgroundColor: '#f8f9fa',
        padding: '2rem',
        borderRadius: '10px',
        marginBottom: '3rem',
        border: '2px solid #e0e0e0'
      }}>
        <h2 style={{ marginTop: 0 }}>🎛️ Voice Settings</h2>
        <p style={{ color: '#666', marginBottom: '1.5rem' }}>
          Customize how your voice sounds. These base settings will be automatically adjusted based on energy levels.
        </p>

        <div style={{ backgroundColor: '#fff', padding: '1.5rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
          <h3 style={{ marginTop: 0, fontSize: '16px' }}>📊 Energy Ranges Explanation</h3>
          <ul style={{ lineHeight: '1.8', marginBottom: 0, fontSize: '14px' }}>
            <li><strong>Low Energy (0-0.33):</strong> Calm, subdued delivery</li>
            <li><strong>Medium Energy (0.34-0.66):</strong> Balanced, natural speech</li>
            <li><strong>High Energy (0.67-1.0):</strong> Expressive, dynamic delivery</li>
            <li><strong>Contradictory (0.1):</strong> Flat, sarcastic tone (fixed at 0.1)</li>
          </ul>
        </div>

        {/* Stability Slider */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <label style={{ fontWeight: 'bold', fontSize: '15px' }}>{tooltips.stability.title}</label>
            <button
              onMouseEnter={() => setShowTooltip('stability')}
              onMouseLeave={() => setShowTooltip(null)}
              style={{
                background: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                width: '18px',
                height: '18px',
                cursor: 'help',
                fontSize: '11px',
                lineHeight: '18px'
              }}
            >
              ?
            </button>
          </div>
          {showTooltip === 'stability' && (
            <div style={{
              backgroundColor: '#333',
              color: 'white',
              padding: '10px',
              borderRadius: '6px',
              marginBottom: '10px',
              fontSize: '13px'
            }}>
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
            style={{ width: '100%', marginBottom: '5px' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#666' }}>
            <span>0.0 (Very Expressive)</span>
            <span style={{ fontWeight: 'bold', color: '#000' }}>{settings.stability.toFixed(2)}</span>
            <span>1.0 (Very Consistent)</span>
          </div>
        </div>

        {/* Similarity Boost Slider */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <label style={{ fontWeight: 'bold', fontSize: '15px' }}>{tooltips.similarity_boost.title}</label>
            <button
              onMouseEnter={() => setShowTooltip('similarity_boost')}
              onMouseLeave={() => setShowTooltip(null)}
              style={{
                background: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                width: '18px',
                height: '18px',
                cursor: 'help',
                fontSize: '11px',
                lineHeight: '18px'
              }}
            >
              ?
            </button>
          </div>
          {showTooltip === 'similarity_boost' && (
            <div style={{
              backgroundColor: '#333',
              color: 'white',
              padding: '10px',
              borderRadius: '6px',
              marginBottom: '10px',
              fontSize: '13px'
            }}>
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
            style={{ width: '100%', marginBottom: '5px' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#666' }}>
            <span>0.0 (Less Similar)</span>
            <span style={{ fontWeight: 'bold', color: '#000' }}>{settings.similarity_boost.toFixed(2)}</span>
            <span>1.0 (More Similar)</span>
          </div>
        </div>

        {/* Style Slider */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <label style={{ fontWeight: 'bold', fontSize: '15px' }}>{tooltips.style.title}</label>
            <button
              onMouseEnter={() => setShowTooltip('style')}
              onMouseLeave={() => setShowTooltip(null)}
              style={{
                background: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                width: '18px',
                height: '18px',
                cursor: 'help',
                fontSize: '11px',
                lineHeight: '18px'
              }}
            >
              ?
            </button>
          </div>
          {showTooltip === 'style' && (
            <div style={{
              backgroundColor: '#333',
              color: 'white',
              padding: '10px',
              borderRadius: '6px',
              marginBottom: '10px',
              fontSize: '13px'
            }}>
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
            style={{ width: '100%', marginBottom: '5px' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#666' }}>
            <span>0.0 (Neutral/Fast)</span>
            <span style={{ fontWeight: 'bold', color: '#000' }}>{settings.style.toFixed(2)}</span>
            <span>1.0 (Exaggerated/Slow)</span>
          </div>
        </div>

        {/* Speaker Boost Toggle */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <label style={{ fontWeight: 'bold', fontSize: '15px' }}>{tooltips.use_speaker_boost.title}</label>
            <button
              onMouseEnter={() => setShowTooltip('use_speaker_boost')}
              onMouseLeave={() => setShowTooltip(null)}
              style={{
                background: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                width: '18px',
                height: '18px',
                cursor: 'help',
                fontSize: '11px',
                lineHeight: '18px'
              }}
            >
              ?
            </button>
          </div>
          {showTooltip === 'use_speaker_boost' && (
            <div style={{
              backgroundColor: '#333',
              color: 'white',
              padding: '10px',
              borderRadius: '6px',
              marginBottom: '10px',
              fontSize: '13px'
            }}>
              {tooltips.use_speaker_boost.description}
            </div>
          )}
          <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={settings.use_speaker_boost}
              onChange={(e) => setSettings({ ...settings, use_speaker_boost: e.target.checked })}
              style={{ width: '18px', height: '18px' }}
            />
            <span style={{ fontSize: '14px' }}>{settings.use_speaker_boost ? 'Enabled (Recommended)' : 'Disabled'}</span>
          </label>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: '10px', marginTop: '1.5rem' }}>
          <button
            onClick={handleSaveSettings}
            disabled={savingSettings}
            style={{
              padding: '10px 20px',
              backgroundColor: savingSettings ? '#6c757d' : '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: savingSettings ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: 'bold'
            }}
          >
            {savingSettings ? '💾 Saving...' : '💾 Save Settings'}
          </button>
          <button
            onClick={handleResetSettings}
            style={{
              padding: '10px 20px',
              backgroundColor: '#dc3545',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            🔄 Reset to Defaults
          </button>
        </div>
      </div>

      {/* Voice Selection Section */}
      <h2>🎤 Available Voices</h2>
      <p style={{ marginBottom: '1.5rem' }}>Preview and select your preferred voice.</p>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: '1.5rem',
        }}
      >
        {voices.map((v) => (
          <div
            key={v.voice_id}
            style={{
              border: '1px solid #ccc',
              borderRadius: '10px',
              padding: '1rem',
              backgroundColor: playingVoiceId === v.voice_id ? '#e3f2fd' : '#fff',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
              transition: 'all 0.2s',
            }}
          >
            <h3>{v.name}</h3>
            <p style={{ color: '#777', fontSize: '0.9rem' }}>
              Category: {v.category || 'Unknown'}
            </p>
            {v.labels && (
              <p style={{ color: '#999', fontSize: '0.8rem' }}>
                {v.labels.accent && `${v.labels.accent}`}
                {v.labels.age && ` • ${v.labels.age}`}
                {v.labels.gender && ` • ${v.labels.gender}`}
              </p>
            )}
            <div style={{ marginTop: '10px' }}>
              <button
                onClick={() => handlePreview(v)}
                disabled={playingVoiceId === v.voice_id}
                style={{
                  backgroundColor: playingVoiceId === v.voice_id ? '#6c757d' : '#007bff',
                  color: 'white',
                  border: 'none',
                  padding: '8px 16px',
                  borderRadius: '5px',
                  cursor: playingVoiceId === v.voice_id ? 'not-allowed' : 'pointer',
                  marginRight: '10px',
                  opacity: playingVoiceId === v.voice_id ? 0.6 : 1,
                }}
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
                    // Keep localStorage for backward compatibility
                    localStorage.setItem('selectedVoice', v.voice_id)
                    localStorage.setItem('selectedVoiceName', v.name)
                    alert(`✅ Selected voice: ${v.name}`)
                  } catch (error) {
                    console.error('❌ Error saving voice selection:', error)
                    alert('❌ Failed to save voice selection')
                  }
                }}
                style={{
                  backgroundColor: selectedVoiceId === v.voice_id ? '#6c757d' : '#28a745',
                  color: 'white',
                  border: 'none',
                  padding: '8px 16px',
                  borderRadius: '5px',
                  cursor: 'pointer',
                }}
              >
                {selectedVoiceId === v.voice_id ? '✓ Selected' : '✅ Select'}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default VoiceDashboard