"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Brain, ArrowLeft, Download, Activity, Moon, Calendar, FileText } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
export default function HistoryPage() {
  const { user, loading: authLoading } = useAuth();
  const [analyses, setAnalyses] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const router = useRouter();
  useEffect(() => {
    if (!authLoading && user) {
      const token = localStorage.getItem("token");
      if (token) {
        fetchHistory(token);
      }
    }
  }, [authLoading, user]);
  const fetchHistory = async (token: string) => {
    try {
      const res = await fetch("/api/history", {
        headers: {
          "Authorization": `Bearer ${token}`
        }
      });
      const data = await res.json();
      if (res.ok) {
        setAnalyses(data.analyses);
      } else {
        setError(data.error || "Failed to load history");
      }
    } catch (err) {
      setError("Network error. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };
  if (authLoading || isLoading || !user) {
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
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Analysis History</h1>
        <p className="text-lg text-gray-600 mb-8">View all your previous EEG analyses</p>
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        ) : error ? (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
            <p className="text-red-700">{error}</p>
          </div>
        ) : analyses.length === 0 ? (
          <div className="bg-white rounded-2xl shadow-xl p-12 text-center border border-gray-100">
            <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-gray-900 mb-2">No Analyses Yet</h3>
            <p className="text-gray-600 mb-6">You haven't performed any EEG analyses yet.</p>
            <Link href="/dashboard">
              <button className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition">
                Start Your First Analysis
              </button>
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {analyses.map((analysis) => (
              <div key={analysis.id} className="bg-white rounded-xl shadow-md p-6 border border-gray-100 hover:shadow-lg transition">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4 flex-1">
                    <div className={`p-3 rounded-lg ${
                      analysis.analysisType === 'seizure' 
                        ? 'bg-red-100' 
                        : 'bg-purple-100'
                    }`}>
                      {analysis.analysisType === 'seizure' ? (
                        <Activity className="w-6 h-6 text-red-600" />
                      ) : (
                        <Moon className="w-6 h-6 text-purple-600" />
                      )}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <h3 className="text-lg font-bold text-gray-900">
                          {analysis.analysisType === 'seizure' 
                            ? 'Seizure Detection' 
                            : 'Sleep Stage Classification'}
                        </h3>
                        <span className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full font-medium">
                          {analysis.fileFormat.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-gray-600 mb-2">
                        <span className="font-semibold">File:</span> {analysis.fileName}
                      </p>
                      <div className="flex items-center gap-6 text-sm text-gray-600">
                        <div className="flex items-center gap-1">
                          <Calendar className="w-4 h-4" />
                          {new Date(analysis.createdAt).toLocaleDateString('en-US', {
                            month: 'short',
                            day: 'numeric',
                            year: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </div>
                      </div>
                      <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-gray-600 mb-1">Prediction</p>
                            <p className="font-bold text-gray-900">{analysis.prediction}</p>
                          </div>
                          {analysis.confidence && (
                            <div>
                              <p className="text-sm text-gray-600 mb-1">Confidence</p>
                              <p className="font-bold text-gray-900">{analysis.confidence.toFixed(2)}%</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                  {analysis.reportUrl && (
                    <a
                      href={analysis.reportUrl}
                      download
                      className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition ml-4"
                    >
                      <Download className="w-4 h-4" />
                      <span>PDF</span>
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
