import React, { useState, useRef, useEffect } from 'react';
import { PaperAirplaneIcon } from '@heroicons/react/24/solid';
import { InformationCircleIcon } from '@heroicons/react/24/outline';

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Sample welcome message
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          id: 'welcome',
          type: 'bot',
          text: 'Hello! I\'m Kotori, your companion for navigating Empty Nest Syndrome. How can I help you today?',
          agent: 'welcome'
        }
      ]);
    }
  }, [messages.length]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {
    // Check if it's a real event or a synthetic one
    if (e && e.preventDefault) {
      e.preventDefault();
    }
    
    if (!input.trim()) return;

    const userMessage = {
      id: Date.now().toString(),
      type: 'user',
      text: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setError(null);

    try {
      // Import the chatService
      const { chatService } = await import('../services/api');
      
      // Call the actual API
      const response = await chatService.sendMessage(input);
      
      const botMessage = {
        id: Date.now().toString(),
        type: 'bot',
        text: response.response,
        agent: response.agent
      };

      setMessages((prev) => [...prev, botMessage]);
      setIsLoading(false);
      
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Sorry, there was an error processing your message. Please try again.');
      setIsLoading(false);
    }
  };



  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl sm:text-4xl font-bold text-primary-800 mb-4">
          Chat with Kotori
        </h1>
        <p className="text-xl text-gray-600">
          Ask questions, share your feelings, or request suggestions about Empty Nest Syndrome.
        </p>
      </div>

      <div className="bg-white rounded-2xl shadow-medium overflow-hidden">
        {/* Chat tips */}
        <div className="bg-primary-50 p-4 border-b border-primary-100">
          <div className="flex items-start">
            <InformationCircleIcon className="h-6 w-6 text-primary-600 mt-0.5 flex-shrink-0" />
            <div className="ml-3">
              <h3 className="text-lg font-medium text-primary-800">Tips for better conversations:</h3>
              <ul className="mt-1 text-primary-700 space-y-1">
                <li>• Ask specific questions about Empty Nest Syndrome</li>
                <li>• Share your feelings if you need emotional support</li>
                <li>• Request practical suggestions for coping strategies</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Chat messages */}
        <div className="p-4 sm:p-6 overflow-y-auto">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`rounded-2xl px-4 py-3 max-w-[95%] shadow-sm ${ // Keep max-w-[95%] for responsiveness
                    message.type === 'user'
                      ? 'bg-primary-100 text-primary-900'
                      : 'bg-white border border-gray-200 text-gray-800'
                  }`}
                >
                  <div className="text-lg whitespace-pre-wrap break-words">{message.text}</div>
                  {message.agent && message.type === 'bot' && (
                    <div className="mt-2 text-sm text-gray-500 italic">
                      Handled by: {message.agent} agent
                    </div>
                  )}

                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="rounded-2xl px-4 py-3 bg-white border border-gray-200 text-gray-500">
                  <div className="flex space-x-2">
                    <div className="h-2 w-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="h-2 w-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    <div className="h-2 w-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '600ms' }}></div>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="flex justify-start">
                <div className="rounded-2xl px-4 py-3 bg-red-50 border border-red-200 text-red-800">
                  <div className="text-lg">{error}</div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
            
          </div>
        </div>

        {/* Input form */}
        <div className="border-t border-gray-200 p-4">
          <form onSubmit={handleSubmit} className="flex space-x-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message here..."
              className="input flex-1"
              disabled={isLoading}
              ref={inputRef}
            />
            <button
              type="submit"
              className="btn-primary px-4"
              disabled={isLoading || !input.trim()}
            >
              <PaperAirplaneIcon className="h-6 w-6" />
              <span className="sr-only">Send</span>
            </button>
          </form>
        </div>
      </div>

      <div className="mt-8 text-center text-gray-600">
        <p>
          Remember, Kotori is an AI assistant and not a substitute for professional help.
          If you're experiencing severe distress, please consult a healthcare professional.
        </p>
      </div>
    </div>
  );
}