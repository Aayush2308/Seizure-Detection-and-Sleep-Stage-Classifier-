"use client";
import { useState } from "react";
import Link from "next/link";
import { Brain, Mail, ArrowLeft } from "lucide-react";
export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [resetUrl, setResetUrl] = useState("");
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage("");
    setResetUrl("");
    try {
      const res = await fetch("/api/forgot-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      const data = await res.json();
      setMessage(data.message);
      if (data.resetUrl) {
        setResetUrl(data.resetUrl);
      }
    } catch (error) {
      setMessage("Network error. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };
  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center px-4">
      <div className="max-w-md w-full">
        <div className="text-center mb-8">
          <Link href="/" className="inline-flex items-center gap-2 mb-4">
            <Brain className="w-10 h-10 text-blue-600" />
            <span className="text-2xl font-bold text-gray-800">NeuroDetect AI</span>
          </Link>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Forgot Password</h1>
          <p className="text-gray-600">Enter your email to receive a password reset link</p>
        </div>
        <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="email"
                  placeholder="Enter your email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition"
                />
              </div>
            </div>
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-linear-to-r from-blue-600 to-purple-600 text-white font-semibold py-3 rounded-lg hover:shadow-lg transition transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
            >
              {isLoading ? "Sending..." : "Send Reset Link"}
            </button>
          </form>
          {message && (
            <div className="mt-4 p-3 rounded-lg text-sm bg-blue-100 text-blue-700">
              {message}
            </div>
          )}
          {resetUrl && (
            <div className="mt-4 p-3 rounded-lg text-sm bg-yellow-100 text-yellow-800">
              <p className="font-semibold mb-2">Development Mode - Reset Link:</p>
              <Link href={resetUrl} className="text-blue-600 hover:underline break-all">
                {resetUrl}
              </Link>
            </div>
          )}
          <div className="mt-6 text-center space-y-2">
            <Link href="/login" className="flex items-center justify-center gap-2 text-blue-600 font-semibold hover:underline">
              <ArrowLeft className="w-4 h-4" />
              <span>Back to Login</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
