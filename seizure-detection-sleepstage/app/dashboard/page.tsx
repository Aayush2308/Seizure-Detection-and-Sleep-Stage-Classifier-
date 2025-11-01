"use client";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Brain, Activity, Moon, LogOut, Settings, UserCircle } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
export default function DashboardPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const handleLogout = async () => {
    const token = localStorage.getItem("token");
    if (token) {
      try {
        await fetch("/api/logout", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token }),
        });
      } catch (error) {
        console.error("Logout error:", error);
      }
    }
    localStorage.removeItem("token");
    router.push("/");
  };
  if (loading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }
  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-white to-purple-50">
      {/* Navigation Bar */}
      <nav className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-2">
              <Brain className="w-8 h-8 text-blue-600" />
              <span className="text-xl font-bold text-gray-800">NeuroDetect AI</span>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <UserCircle className="w-5 h-5 text-gray-600" />
                <span className="text-gray-700 font-medium">{user.name}</span>
              </div>
              <Link href="/settings">
                <button className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:bg-gray-50 rounded-lg transition">
                  <Settings className="w-5 h-5" />
                  <span>Settings</span>
                </button>
              </Link>
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg transition"
              >
                <LogOut className="w-5 h-5" />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </nav>
      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Welcome Section */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Welcome back, {user.name}! 👋
          </h1>
          <p className="text-xl text-gray-600">
            Choose an analysis tool to get started with your EEG data
          </p>
        </div>
        {/* Analysis Options */}
        <div className="grid md:grid-cols-2 gap-8">
          {/* Seizure Detection Card */}
          <Link href="/seizure-detection">
            <div className="bg-white rounded-2xl shadow-xl p-8 border-2 border-gray-100 hover:border-red-300 hover:shadow-2xl transition transform hover:scale-105 cursor-pointer group">
              <div className="flex items-center gap-4 mb-4">
                <div className="p-4 bg-red-100 rounded-xl group-hover:bg-red-200 transition">
                  <Activity className="w-10 h-10 text-red-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900">Seizure Detection</h2>
              </div>
              <p className="text-gray-600 mb-6">
                Upload EEG data to detect epileptic seizures using our advanced machine learning model. 
                Get instant predictions with detailed analysis reports.
              </p>
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Real-time seizure detection</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Detailed PDF reports</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Multiple file formats supported</span>
                </div>
              </div>
              <div className="mt-6 flex items-center text-red-600 font-semibold group-hover:gap-3 gap-2 transition-all">
                <span>Start Analysis</span>
                <span className="text-xl">→</span>
              </div>
            </div>
          </Link>
          {/* Sleep Stage Classification Card */}
          <Link href="/sleep-stage">
            <div className="bg-white rounded-2xl shadow-xl p-8 border-2 border-gray-100 hover:border-purple-300 hover:shadow-2xl transition transform hover:scale-105 cursor-pointer group">
              <div className="flex items-center gap-4 mb-4">
                <div className="p-4 bg-purple-100 rounded-xl group-hover:bg-purple-200 transition">
                  <Moon className="w-10 h-10 text-purple-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900">Sleep Stage Classification</h2>
              </div>
              <p className="text-gray-600 mb-6">
                Analyze sleep patterns and classify different sleep stages (Wake, REM, NREM) 
                to understand sleep quality and architecture.
              </p>
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>Multi-stage classification</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>Sleep quality insights</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>Comprehensive visualization</span>
                </div>
              </div>
              <div className="mt-6 flex items-center text-purple-600 font-semibold group-hover:gap-3 gap-2 transition-all">
                <span>Start Analysis</span>
                <span className="text-xl">→</span>
              </div>
            </div>
          </Link>
        </div>
        {/* Info Section */}
        <div className="mt-12 bg-blue-50 rounded-2xl p-8 border border-blue-100">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Supported File Formats</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-lg p-4 text-center shadow-sm">
              <p className="font-bold text-blue-600 text-lg">.EDF</p>
              <p className="text-xs text-gray-600">European Data Format</p>
            </div>
            <div className="bg-white rounded-lg p-4 text-center shadow-sm">
              <p className="font-bold text-purple-600 text-lg">.NPZ</p>
              <p className="text-xs text-gray-600">NumPy Archive</p>
            </div>
            <div className="bg-white rounded-lg p-4 text-center shadow-sm">
              <p className="font-bold text-green-600 text-lg">.CSV</p>
              <p className="text-xs text-gray-600">Comma-Separated Values</p>
            </div>
            <div className="bg-white rounded-lg p-4 text-center shadow-sm">
              <p className="font-bold text-orange-600 text-lg">.PKL</p>
              <p className="text-xs text-gray-600">Pickle Format</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
