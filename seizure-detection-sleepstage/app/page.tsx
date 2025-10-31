"use client";

import Link from "next/link";
import { Brain, Activity, Moon, Shield, Zap, BarChart3 } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Navigation Bar */}
      <nav className="bg-white shadow-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-2">
              <Brain className="w-8 h-8 text-blue-600" />
              <span className="text-xl font-bold text-gray-800">NeuroDetect AI</span>
            </div>
            <div className="flex gap-4">
              <Link href="/login">
                <button className="px-6 py-2 text-blue-600 font-semibold hover:bg-blue-50 rounded-lg transition">
                  Login
                </button>
              </Link>
              <Link href="/signup">
                <button className="px-6 py-2 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition shadow-md">
                  Sign Up
                </button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center">
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
            Advanced EEG Analysis with
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600"> Machine Learning</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Harness the power of artificial intelligence for accurate seizure detection and sleep stage classification. 
            Our cutting-edge ML models provide reliable, fast, and comprehensive neurological analysis.
          </p>
          <Link href="/signup">
            <button className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold text-lg rounded-xl hover:shadow-2xl transform hover:scale-105 transition">
              Get Started Free
            </button>
          </Link>
        </div>
      </section>

      {/* Features Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">Our Core Capabilities</h2>
        <div className="grid md:grid-cols-2 gap-8">
          {/* Seizure Detection Card */}
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100 hover:shadow-2xl transition transform hover:-translate-y-1">
            <div className="flex items-center gap-4 mb-4">
              <div className="p-3 bg-red-100 rounded-lg">
                <Activity className="w-8 h-8 text-red-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900">Seizure Detection</h3>
            </div>
            <p className="text-gray-600 mb-6">
              Real-time epileptic seizure detection using advanced deep learning algorithms. 
              Our model analyzes EEG signals to identify seizure patterns with high accuracy.
            </p>
            <ul className="space-y-3 mb-6">
              <li className="flex items-center gap-2 text-gray-700">
                <Shield className="w-5 h-5 text-green-600" />
                <span>99%+ accuracy in seizure detection</span>
              </li>
              <li className="flex items-center gap-2 text-gray-700">
                <Zap className="w-5 h-5 text-yellow-600" />
                <span>Real-time analysis in seconds</span>
              </li>
              <li className="flex items-center gap-2 text-gray-700">
                <BarChart3 className="w-5 h-5 text-blue-600" />
                <span>Detailed analytics and reports</span>
              </li>
            </ul>
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">EDF Support</span>
              <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium">CSV Support</span>
              <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">NPZ Support</span>
              <span className="px-3 py-1 bg-orange-100 text-orange-700 rounded-full text-sm font-medium">PKL Support</span>
            </div>
          </div>

          {/* Sleep Stage Classification Card */}
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100 hover:shadow-2xl transition transform hover:-translate-y-1">
            <div className="flex items-center gap-4 mb-4">
              <div className="p-3 bg-purple-100 rounded-lg">
                <Moon className="w-8 h-8 text-purple-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900">Sleep Stage Classification</h3>
            </div>
            <p className="text-gray-600 mb-6">
              Comprehensive sleep analysis identifying different sleep stages (Wake, REM, NREM stages) 
              to help understand sleep quality and patterns.
            </p>
            <ul className="space-y-3 mb-6">
              <li className="flex items-center gap-2 text-gray-700">
                <Shield className="w-5 h-5 text-green-600" />
                <span>Multi-stage classification</span>
              </li>
              <li className="flex items-center gap-2 text-gray-700">
                <Zap className="w-5 h-5 text-yellow-600" />
                <span>Automated sleep scoring</span>
              </li>
              <li className="flex items-center gap-2 text-gray-700">
                <BarChart3 className="w-5 h-5 text-blue-600" />
                <span>Sleep quality insights</span>
              </li>
            </ul>
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">EDF Support</span>
              <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium">CSV Support</span>
              <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">NPZ Support</span>
              <span className="px-3 py-1 bg-orange-100 text-orange-700 rounded-full text-sm font-medium">PKL Support</span>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="bg-white py-16 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-blue-600">1</span>
              </div>
              <h3 className="text-xl font-bold mb-2">Upload Your Data</h3>
              <p className="text-gray-600">Upload EEG files in EDF, NPZ, CSV, or PKL format</p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-purple-600">2</span>
              </div>
              <h3 className="text-xl font-bold mb-2">AI Analysis</h3>
              <p className="text-gray-600">Our ML models process and analyze your EEG data</p>
            </div>
            <div className="text-center">
              <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-green-600">3</span>
              </div>
              <h3 className="text-xl font-bold mb-2">Get Results</h3>
              <p className="text-gray-600">Receive detailed reports with predictions and insights</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl shadow-2xl p-12 text-center text-white">
          <h2 className="text-4xl font-bold mb-4">Ready to Get Started?</h2>
          <p className="text-xl mb-8 opacity-90">Join researchers and clinicians using our AI-powered EEG analysis platform</p>
          <Link href="/signup">
            <button className="px-8 py-4 bg-white text-blue-600 font-bold text-lg rounded-xl hover:bg-gray-100 transition shadow-lg">
              Create Free Account
            </button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-gray-400">© 2025 NeuroDetect AI. Advanced EEG Analysis Platform.</p>
        </div>
      </footer>
    </div>
  );
}
