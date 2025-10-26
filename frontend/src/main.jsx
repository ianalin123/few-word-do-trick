import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import VoiceDashboard from './components/VoiceDashboard'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/voices" element={<VoiceDashboard />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
)
