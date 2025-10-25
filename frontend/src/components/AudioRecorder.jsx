import React, { useState, useRef, useEffect } from 'react'

const AudioRecorder = ({ onTranscription }) => {
  const [mediaRecorder, setMediaRecorder] = useState(null)
  const [audioChunks, setAudioChunks] = useState([])
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [audioLevel, setAudioLevel] = useState(0)
  const [hasPermission, setHasPermission] = useState(null)
  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const animationFrameRef = useRef(null)

  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
      }
    }
  }, [])

  const startRecording = async () => {
    try {
      console.log('Requesting microphone access...')
      
      // Check if we're on HTTPS or localhost
      if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
        alert('Microphone access requires HTTPS. Please use https:// or localhost for development.')
        return
      }
      
      // Request microphone permission with more specific error handling
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
          channelCount: 1
        } 
      })
      
      console.log('Microphone access granted, starting recording...')
      setHasPermission(true)
      
      const recorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })
      
      const chunks = []
      
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data)
          console.log('Audio data received:', event.data.size, 'bytes')
        }
      }
      
      recorder.onstop = async () => {
        console.log('Recording stopped, processing audio...')
        const audioBlob = new Blob(chunks, { type: 'audio/webm' })
        await processAudio(audioBlob)
        stream.getTracks().forEach(track => track.stop())
      }
      
      recorder.onstart = () => {
        console.log('Recording started')
        setIsRecording(true)
        console.log('isRecording state set to true')
      }
      
      recorder.onerror = (event) => {
        console.error('Recording error:', event.error)
        setIsRecording(false)
      }
      
      // Set up audio visualization
      setupAudioVisualization(stream)
      
      setMediaRecorder(recorder)
      setAudioChunks(chunks)
      recorder.start(100) // Collect data every 100ms
      
    } catch (error) {
      console.error('Error accessing microphone:', error)
      let errorMessage = 'Microphone access denied. '
      
      if (error.name === 'NotAllowedError') {
        errorMessage += 'Please allow microphone access in your browser settings and refresh the page.'
      } else if (error.name === 'NotFoundError') {
        errorMessage += 'No microphone found. Please connect a microphone.'
      } else if (error.name === 'NotSupportedError') {
        errorMessage += 'Your browser does not support audio recording.'
      } else {
        errorMessage += 'Please check your microphone and try again.'
      }
      
      console.error('Microphone error:', error)
      alert(errorMessage)
      setHasPermission(false)
      setIsRecording(false)
    }
  }

  const setupAudioVisualization = (stream) => {
    try {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)()
      const source = audioContextRef.current.createMediaStreamSource(stream)
      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 256
      analyserRef.current.smoothingTimeConstant = 0.8
      source.connect(analyserRef.current)
      
      const updateAudioLevel = () => {
        if (analyserRef.current && isRecording) {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
          analyserRef.current.getByteFrequencyData(dataArray)
          
          // Calculate average volume
          const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length
          setAudioLevel(average)
          
          // Continue animation while recording
          if (isRecording) {
            animationFrameRef.current = requestAnimationFrame(updateAudioLevel)
          }
        }
      }
      updateAudioLevel()
    } catch (error) {
      console.error('Error setting up audio visualization:', error)
    }
  }

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      console.log('Stopping recording...')
      mediaRecorder.stop()
      setIsRecording(false)
      setAudioLevel(0)
    }
  }

  const processAudio = async (audioBlob) => {
    setIsProcessing(true)
    try {
      const formData = new FormData()
      formData.append('audio_file', audioBlob, 'recording.webm')
      
      const response = await fetch('/api/speech-to-text', {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        throw new Error('Speech-to-text failed')
      }
      
      const result = await response.json()
      onTranscription(result.text)
      
    } catch (error) {
      console.error('Error processing audio:', error)
      alert('Failed to process audio. Please try again.')
    } finally {
      setIsProcessing(false)
    }
  }

  const getEmoji = () => {
    if (isProcessing) return '🔄'
    if (isRecording) return '⏹️'
    return '🎤'
  }

  const getButtonText = () => {
    if (isProcessing) return 'Processing...'
    if (isRecording) return 'Stop Recording'
    return 'Start Recording'
  }

  const getStatusText = () => {
    if (isProcessing) return 'Processing audio...'
    if (isRecording) return '🔴 Recording... Click to stop'
    if (hasPermission === false) return '❌ Microphone access denied'
    return 'Click to start recording'
  }

  return (
    <div className="audio-recorder">
      <h3>Voice Input</h3>
      
      <button
        className={`record-button ${isRecording ? 'recording' : ''}`}
        onClick={isRecording ? stopRecording : startRecording}
        disabled={isProcessing}
        title={getButtonText()}
      >
        {getEmoji()}
      </button>
      
      <div className="audio-visualizer">
        {Array.from({ length: 5 }, (_, i) => (
          <div
            key={i}
            className="audio-bar"
            style={{
              height: isRecording ? `${10 + (audioLevel / 4)}px` : '10px',
              opacity: isRecording ? 1 : 0.3
            }}
          />
        ))}
      </div>
      
      <p>
        {getStatusText()}
      </p>
      
      {isRecording && (
        <div className="recording-indicator">
          <div className="pulse-dot"></div>
          <span>Recording in progress...</span>
        </div>
      )}
    </div>
  )
}

export default AudioRecorder
