import React from 'react';
import { Link } from 'react-router-dom';
import { ChatBubbleLeftRightIcon, HeartIcon, LightBulbIcon } from '@heroicons/react/24/outline';

export default function About() {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-primary-800 mb-6">
          About Kotori.ai
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Your supportive companion for navigating Empty Nest Syndrome.
        </p>
      </div>

      <div className="bg-white rounded-2xl shadow-soft overflow-hidden mb-12">
        <div className="p-8">
          <h2 className="text-2xl font-semibold text-primary-800 mb-6">Our Mission</h2>
          <p className="text-lg text-gray-600 mb-6">
            Kotori.ai was created to provide support, information, and guidance to parents 
            experiencing Empty Nest Syndrome. We understand that when children leave home, 
            it can trigger a range of emotions and challenges for parents.
          </p>
          <p className="text-lg text-gray-600 mb-6">
            Our mission is to help you navigate this significant life transition with 
            compassion, practical advice, and evidence-based information. We believe that 
            this new chapter in your life can be fulfilling and meaningful with the right 
            support and perspective.
          </p>
          <p className="text-lg text-gray-600">
            Whether you're seeking information about Empty Nest Syndrome, emotional support 
            during difficult moments, or practical suggestions for moving forward, Kotori 
            is here to help you every step of the way.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
        <div className="bg-white rounded-2xl shadow-soft p-6 flex flex-col items-center text-center">
          <ChatBubbleLeftRightIcon className="h-12 w-12 text-primary-600 mb-4" />
          <h3 className="text-xl font-semibold text-primary-700 mb-3">Information</h3>
          <p className="text-gray-600">
            Kotori provides accurate, evidence-based information about Empty Nest Syndrome, 
            its symptoms, and common experiences.
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-soft p-6 flex flex-col items-center text-center">
          <HeartIcon className="h-12 w-12 text-primary-600 mb-4" />
          <h3 className="text-xl font-semibold text-primary-700 mb-3">Emotional Support</h3>
          <p className="text-gray-600">
            When you're feeling sad, lonely, or overwhelmed, Kotori offers compassionate 
            emotional support and validation.
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-soft p-6 flex flex-col items-center text-center">
          <LightBulbIcon className="h-12 w-12 text-primary-600 mb-4" />
          <h3 className="text-xl font-semibold text-primary-700 mb-3">Practical Suggestions</h3>
          <p className="text-gray-600">
            Kotori provides actionable suggestions and strategies to help you adapt, grow, 
            and thrive during this new life phase.
          </p>
        </div>
      </div>

      <div className="bg-white rounded-2xl shadow-soft overflow-hidden mb-12">
        <div className="p-8">
          <h2 className="text-2xl font-semibold text-primary-800 mb-6">How Kotori Works</h2>
          <p className="text-lg text-gray-600 mb-6">
            Kotori uses advanced artificial intelligence to understand your questions and concerns 
            about Empty Nest Syndrome. Our system is designed to provide three types of support:
          </p>
          
          <div className="space-y-4 mb-6">
            <div className="bg-gray-50 rounded-xl p-4">
              <h3 className="text-lg font-medium text-primary-700 mb-2">Question Answering</h3>
              <p className="text-gray-600">
                When you ask for information about Empty Nest Syndrome, Kotori draws on a 
                knowledge base of reliable resources to provide accurate answers.
              </p>
            </div>
            
            <div className="bg-gray-50 rounded-xl p-4">
              <h3 className="text-lg font-medium text-primary-700 mb-2">Emotional Check-ins</h3>
              <p className="text-gray-600">
                When you express feelings of sadness, loneliness, or other emotions, Kotori 
                responds with empathy and supportive guidance.
              </p>
            </div>
            
            <div className="bg-gray-50 rounded-xl p-4">
              <h3 className="text-lg font-medium text-primary-700 mb-2">Suggestions & Advice</h3>
              <p className="text-gray-600">
                When you ask for help or suggestions, Kotori offers practical, actionable 
                advice to help you navigate this transition.
              </p>
            </div>
          </div>
          
          <p className="text-lg text-gray-600">
            While Kotori is designed to be helpful and supportive, it's important to remember 
            that it's not a substitute for professional mental health support. If you're 
            experiencing severe distress, please consult with a healthcare professional.
          </p>
        </div>
      </div>

      <div className="bg-primary-50 rounded-2xl p-8 text-center">
        <h2 className="text-2xl font-semibold text-primary-800 mb-4">
          Ready to Connect with Kotori?
        </h2>
        <p className="text-lg text-gray-600 mb-6 max-w-3xl mx-auto">
          Whether you have questions, need emotional support, or are looking for practical 
          suggestions, Kotori is here to help you navigate Empty Nest Syndrome.
        </p>
        <Link to="/chat" className="btn-primary text-lg px-8">
          Start Chatting Now
        </Link>
      </div>
    </div>
  );
}