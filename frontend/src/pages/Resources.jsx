import React from 'react';
import { Link } from 'react-router-dom';
import { BookOpenIcon, DocumentTextIcon, UserGroupIcon, AcademicCapIcon } from '@heroicons/react/24/outline';

export default function Resources() {
  const resourceCategories = [
    {
      title: 'Articles & Guides',
      icon: DocumentTextIcon,
      resources: [
        {
          title: 'Understanding Empty Nest Syndrome',
          description: 'A comprehensive guide to the emotional journey of Empty Nest Syndrome.',
          link: '#',
        },
        {
          title: 'Coping Strategies for Parents',
          description: 'Practical approaches to managing the transition when children leave home.',
          link: '#',
        },
        {
          title: 'Rediscovering Your Identity',
          description: 'How to reconnect with yourself and find new purpose after active parenting.',
          link: '#',
        },
      ],
    },
    {
      title: 'Support Groups',
      icon: UserGroupIcon,
      resources: [
        {
          title: 'Empty Nesters Community',
          description: 'An online forum for parents experiencing or navigating Empty Nest Syndrome.',
          link: '#',
        },
        {
          title: 'Local Support Meetings',
          description: 'Find in-person support groups in your area for shared experiences.',
          link: '#',
        },
        {
          title: 'Parent-to-Parent Mentoring',
          description: 'Connect with parents who have successfully navigated this transition.',
          link: '#',
        },
      ],
    },
    {
      title: 'Books & Reading',
      icon: BookOpenIcon,
      resources: [
        {
          title: 'Empty Nesting: Reinventing Your Life',
          description: 'A guide to finding new meaning and joy in the empty nest phase.',
          link: '#',
        },
        {
          title: 'The Changing Family Dynamic',
          description: 'Understanding and embracing new family relationships after children leave.',
          link: '#',
        },
        {
          title: 'From Parents to Partners',
          description: 'Reconnecting with your spouse or partner after children leave home.',
          link: '#',
        },
      ],
    },
    {
      title: 'Educational Resources',
      icon: AcademicCapIcon,
      resources: [
        {
          title: 'The Psychology of Empty Nest Syndrome',
          description: 'Research-based insights into the emotional and psychological aspects.',
          link: '#',
        },
        {
          title: 'Webinars & Workshops',
          description: 'Online learning opportunities focused on this life transition.',
          link: '#',
        },
        {
          title: 'Expert Interviews',
          description: 'Conversations with psychologists and counselors specializing in family transitions.',
          link: '#',
        },
      ],
    },
  ];

  return (
    <div className="max-w-5xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-primary-800 mb-6">
          Helpful Resources
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Explore these carefully selected resources to help you navigate Empty Nest Syndrome 
          and embrace this new chapter in your life.
        </p>
      </div>

      <div className="space-y-12">
        {resourceCategories.map((category, index) => (
          <div key={index} className="bg-white rounded-2xl shadow-soft overflow-hidden">
            <div className="bg-primary-50 p-6 border-b border-primary-100">
              <div className="flex items-center">
                <category.icon className="h-8 w-8 text-primary-600" />
                <h2 className="ml-3 text-2xl font-semibold text-primary-800">{category.title}</h2>
              </div>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {category.resources.map((resource, resourceIndex) => (
                  <div key={resourceIndex} className="border border-gray-200 rounded-xl p-5 hover:border-primary-300 hover:shadow-soft transition-all duration-200">
                    <h3 className="text-xl font-medium text-primary-700 mb-2">{resource.title}</h3>
                    <p className="text-gray-600 mb-4">{resource.description}</p>
                    <a 
                      href={resource.link} 
                      className="text-primary-600 font-medium hover:text-primary-800 transition-colors"
                    >
                      Learn more →
                    </a>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-primary-50 rounded-2xl p-8 mt-12 text-center">
        <h2 className="text-2xl font-semibold text-primary-800 mb-4">
          Need Personalized Support?
        </h2>
        <p className="text-lg text-gray-600 mb-6 max-w-3xl mx-auto">
          If you'd like more personalized guidance or have specific questions about Empty Nest Syndrome,
          Kotori is here to help.
        </p>
        <Link to="/chat" className="btn-primary text-lg px-8">
          Chat with Kotori
        </Link>
      </div>
    </div>
  );
}