import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3001,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    },
    allowedHosts: ['localhost', '127.0.0.1', '0.0.0.0', 'conjoinedly-snapless-sid.ngrok-free.dev']
  }
})