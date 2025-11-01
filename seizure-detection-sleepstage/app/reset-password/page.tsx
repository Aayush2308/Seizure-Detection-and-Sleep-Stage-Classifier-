"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Brain, Lock } from "lucide-react";
export default function ResetPasswordPage() {
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [token, setToken] = useState("");
  const router = useRouter();
  const searchParams = useSearchParams();
  useEffect(() => {
    const tokenParam = searchParams.get("token");
    if (tokenParam) {
      setToken(tokenParam);
    } else {
      setMessage("Invalid reset link. Please request a new password reset.");
    }
  }, [searchParams]);
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      setMessage("Passwords do not match");
      return;
    }
    if (password.length < 6) {
      setMessage("Password must be at least 6 characters");
      return;
    }
    setIsLoading(true);
    setMessage("");
    try {
      const res = await fetch("/api/reset-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, password }),
      });
      const data = await res.json();
      if (res.ok) {
        setMessage(data.message);
        setTimeout(() => router.push("/login"), 2000);
      } else {
        setMessage(data.error);
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
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Reset Password</h1>
          <p className="text-gray-600">Enter your new password</p>
        </div>
        <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                New Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="password"
                  placeholder="Enter new password"
                  required
                  minLength={6}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition"
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Confirm Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="password"
                  placeholder="Confirm new password"
                  required
                  minLength={6}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition"
                />
              </div>
            </div>
            <button
              type="submit"
              disabled={isLoading || !token}
              className="w-full bg-linear-to-r from-blue-600 to-purple-600 text-white font-semibold py-3 rounded-lg hover:shadow-lg transition transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
            >
              {isLoading ? "Resetting..." : "Reset Password"}
            </button>
          </form>
          {message && (
            <div
              className={`mt-4 p-3 rounded-lg text-sm ${
                message.includes("success")
                  ? "bg-green-100 text-green-700"
                  : "bg-red-100 text-red-700"
              }`}
            >
              {message}
            </div>
          )}
          <div className="mt-6 text-center">
            <Link href="/login" className="text-blue-600 font-semibold hover:underline">
              Back to Login
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
