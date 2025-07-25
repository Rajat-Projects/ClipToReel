import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173, // Frontend runs on this port
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000', // Flask backend
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
