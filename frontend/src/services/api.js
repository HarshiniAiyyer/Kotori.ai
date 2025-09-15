import axios from 'axios';

// Create an axios instance with default config
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 seconds
  withCredentials: false
});

// API functions for Kotori chat
export const chatService = {
  // Send a message to Kotori and get a response
  sendMessage: async (message) => {
    try {
      const response = await api.post('/chat', { input: message });
      return response.data;
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  },

  // Get chat history (if implemented in backend)
  getChatHistory: async () => {
    try {
      const response = await api.get('/chat/history');
      return response.data;
    } catch (error) {
      console.error('Error fetching chat history:', error);
      throw error;
    }
  },

  // Clear chat history (if implemented in backend)
  clearChatHistory: async () => {
    try {
      const response = await api.delete('/chat/history');
      return response.data;
    } catch (error) {
      console.error('Error clearing chat history:', error);
      throw error;
    }
  },
};

export default api;