"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Brain, ArrowLeft, Trash2, History, User as UserIcon } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
export default function SettingsPage() {
  const { user, loading: authLoading } = useAuth();
  const [password, setPassword] = useState("");
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [message, setMessage] = useState("");
  const router = useRouter();
  const handleDeleteAccount = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsDeleting(true);
    setMessage("");
    const token = localStorage.getItem("token");
    try {
      const res = await fetch("/api/delete-account", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ password }),
      });
      const data = await res.json();
      if (res.ok) {
        setMessage("Account deleted successfully. Redirecting...");
        localStorage.removeItem("token");
        setTimeout(() => router.push("/"), 2000);
      } else {
        setMessage(data.error || "Failed to delete account");
      }
    } catch (error) {
      setMessage("Network error. Please try again.");
    } finally {
      setIsDeleting(false);
    }
  };
  if (authLoading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }
  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-white to-purple-50">
      <nav className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-2">
              <Brain className="w-8 h-8 text-blue-600" />
              <span className="text-xl font-bold text-gray-800">NeuroDetect AI</span>
            </div>
            <Link href="/dashboard">
              <button className="flex items-center gap-2 px-4 py-2 text-blue-600 hover:bg-blue-50 rounded-lg transition">
                <ArrowLeft className="w-5 h-5" />
                <span>Back to Dashboard</span>
              </button>
            </Link>
          </div>
        </div>
      </nav>
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-8">Account Settings</h1>
        {/* User Info */}
        <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100 mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
            <UserIcon className="w-6 h-6 text-blue-600" />
            Profile Information
          </h2>
          <div className="space-y-4">
            <div>
              <label className="text-sm font-semibold text-gray-600">Name</label>
              <p className="text-lg text-gray-900">{user.name}</p>
            </div>
            <div>
              <label className="text-sm font-semibold text-gray-600">Email</label>
              <p className="text-lg text-gray-900">{user.email}</p>
            </div>
          </div>
        </div>
        {/* Analysis History Link */}
        <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100 mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            <History className="w-6 h-6 text-blue-600" />
            Analysis History
          </h2>
          <p className="text-gray-600 mb-4">View all your previous EEG analyses and download reports</p>
          <Link href="/history">
            <button className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition">
              View History
            </button>
          </Link>
        </div>
        {/* Delete Account */}
        <div className="bg-white rounded-2xl shadow-xl p-8 border-2 border-red-200">
          <h2 className="text-2xl font-bold text-red-600 mb-4 flex items-center gap-2">
            <Trash2 className="w-6 h-6" />
            Delete Account
          </h2>
          <p className="text-gray-600 mb-6">
            Once you delete your account, there is no going back. All your data, including analysis history and reports, will be permanently deleted.
          </p>
          {!showDeleteConfirm ? (
            <button
              onClick={() => setShowDeleteConfirm(true)}
              className="px-6 py-3 bg-red-600 text-white font-semibold rounded-lg hover:bg-red-700 transition"
            >
              Delete My Account
            </button>
          ) : (
            <form onSubmit={handleDeleteAccount} className="space-y-4">
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-800 font-semibold mb-2">⚠️ This action cannot be undone!</p>
                <p className="text-red-700 text-sm">Please enter your password to confirm account deletion.</p>
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Confirm Password
                </label>
                <input
                  type="password"
                  placeholder="Enter your password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent outline-none transition"
                />
              </div>
              <div className="flex gap-4">
                <button
                  type="button"
                  onClick={() => {
                    setShowDeleteConfirm(false);
                    setPassword("");
                    setMessage("");
                  }}
                  className="px-6 py-3 bg-gray-200 text-gray-700 font-semibold rounded-lg hover:bg-gray-300 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isDeleting}
                  className="px-6 py-3 bg-red-600 text-white font-semibold rounded-lg hover:bg-red-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isDeleting ? "Deleting..." : "Permanently Delete Account"}
                </button>
              </div>
            </form>
          )}
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
        </div>
      </div>
    </div>
  );
}
