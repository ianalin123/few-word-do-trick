import React, { useEffect, useState } from 'react'
import axios from 'axios'

function VoiceDashboard() {
  const [voices, setVoices] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [playingVoiceId, setPlayingVoiceId] = useState(null)

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
    fetchVoices()
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

  if (loading) return <p>Loading voices...</p>
  if (error) return <p style={{ color: 'red' }}>{error}</p>

  return (
    <div style={{ padding: '2rem' }}>
      <h1>🎙️ ElevenLabs Voices</h1>
      <p>Select a voice to preview and choose your default.</p>

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
                onClick={() => {
                  localStorage.setItem('selectedVoice', v.voice_id)
                  localStorage.setItem('selectedVoiceName', v.name)
                  alert(`✅ Selected voice: ${v.name}`)
                }}
                style={{
                  backgroundColor: '#28a745',
                  color: 'white',
                  border: 'none',
                  padding: '8px 16px',
                  borderRadius: '5px',
                  cursor: 'pointer',
                }}
              >
                ✅ Select
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default VoiceDashboard