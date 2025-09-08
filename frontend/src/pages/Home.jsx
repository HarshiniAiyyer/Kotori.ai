import React from 'react';
import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="max-w-5xl mx-auto">
      <div className="text-center mb-16 mt-20">
        <h1 className="text-4xl sm:text-5xl font-bold text-primary-800 mb-6">
          Hi! I am Kotori.
        </h1>
      </div>

      <div className="bg-blue-100 rounded-3xl p-8 mb-16 text-center">
        <h2 className="text-3xl sm:text-4xl font-bold text-blue-900 mb-4">
          Finding it hard to adjust to life with your children gone?
        </h2>
        <p className="text-2xl text-blue-800 max-w-3xl mx-auto">
          You're not alone - many parents experience Empty Nest Syndrome.
        </p>
        <div className="mt-6">
          <p className="text-xl text-blue-700">
            Let Kotori.ai help you navigate this transition.
          </p>
          <div className="mt-4 bg-white rounded-xl p-4 max-w-2xl mx-auto shadow-md">
            <Link to="/chat" className="block text-left text-lg text-gray-700 hover:text-primary-700">
              Talk to Kotori today.
            </Link>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mb-16">
        <div className="card-hover flex flex-col">
          <div className="flex-1">
            <h2 className="text-2xl font-semibold text-primary-700 mb-4">What is Empty Nest Syndrome?</h2>
            <p className="text-lg text-gray-600 mb-6">
              A mix of emotions when children leave home.
            </p>
          </div>
          <Link to="/chat" className="btn-primary w-full text-center">
            Learn More
          </Link>
        </div>

        <div className="card-hover flex flex-col">
          <div className="flex-1">
            <h2 className="text-2xl font-semibold text-primary-700 mb-4">How Kotori Can Help</h2>
            <p className="text-lg text-gray-600 mb-6">
              Personalized support for your empty nest journey.
            </p>
          </div>
          <Link to="/chat" className="btn-primary w-full text-center">
            Chat with Kotori
          </Link>
        </div>
      </div>

      <div className="bg-primary-50 rounded-3xl p-8 mb-16">
        <h2 className="text-3xl font-semibold text-primary-800 mb-6 text-center">
          You're Not Alone.
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-2xl p-6 shadow-soft">
            <h3 className="text-xl font-medium text-primary-700 mb-3">Common Feelings</h3>
            <ul className="text-gray-600 space-y-2">
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">•</span>
                <span>Sadness</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">•</span>
                <span>Purpose loss</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">•</span>
                <span>Worry</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">•</span>
                <span>Identity shift</span>
              </li>
            </ul>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-soft">
            <h3 className="text-xl font-medium text-primary-700 mb-3">Coping Strategies</h3>
            <ul className="text-gray-600 space-y-2">
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">•</span>
                <span>New routines</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">•</span>
                <span>Reconnect</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">•</span>
                <span>New hobbies</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">•</span>
                <span>Self-care</span>
              </li>
            </ul>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-soft">
            <h3 className="text-xl font-medium text-primary-700 mb-3">New Opportunities</h3>
            <ul className="text-gray-600 space-y-2">
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">•</span>
                <span>Self-discovery</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">•</span>
                <span>Relationships</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">•</span>
                <span>Dreams</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">•</span>
                <span>Purpose</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <div className="text-center mb-16">
        <h2 className="text-3xl font-semibold text-primary-800 mb-6">
          Begin Your Next Chapter Today
        </h2>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
          Kotori is here to guide you toward a fulfilling new phase of life.
        </p>
        <Link to="/chat" className="btn-primary text-xl px-10 py-4">
          Start a Conversation
        </Link>
      </div>
    </div>
  );
}