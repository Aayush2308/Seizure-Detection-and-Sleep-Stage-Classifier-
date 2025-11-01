"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Brain, Upload, Activity, ArrowLeft, Download, FileCheck, Loader2 } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
export default function SeizureDetectionPage() {
  const { user, loading } = useAuth();
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");
  const router = useRouter();
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      const validExtensions = ['.edf', '.npz', '.csv', '.pkl'];
      const fileExtension = selectedFile.name.toLowerCase().slice(selectedFile.name.lastIndexOf('.'));
      if (validExtensions.includes(fileExtension)) {
        setFile(selectedFile);
        setError("");
        setResult(null);
      } else {
        setError("Invalid file format. Please upload .edf, .npz, .csv, or .pkl files.");
        setFile(null);
      }
    }
  };
  const handleAnalysis = async () => {
    if (!file) {
      setError("Please select a file first.");
      return;
    }
    setIsAnalyzing(true);
    setError("");
    setResult(null);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("analysisType", "seizure");
    const token = localStorage.getItem("token");
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`
        },
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setResult(data);
      } else {
        setError(data.error || "Analysis failed. Please try again.");
      }
    } catch (err) {
      setError("Network error. Please check your connection and try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };
  if (loading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }
  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-white to-purple-50">
      {/* Navigation */}
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
      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Activity className="w-12 h-12 text-red-600" />
            <h1 className="text-4xl font-bold text-gray-900">Seizure Detection</h1>
          </div>
          <p className="text-lg text-gray-600">
            Upload your EEG data file for automated seizure detection analysis
          </p>
        </div>
        {/* Upload Section */}
        <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100 mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Upload EEG File</h2>
          <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-blue-400 transition">
            <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <label htmlFor="file-upload" className="cursor-pointer">
              <span className="text-blue-600 font-semibold hover:underline">Click to upload</span>
              <span className="text-gray-600"> or drag and drop</span>
              <input
                id="file-upload"
                type="file"
                accept=".edf,.npz,.csv,.pkl"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
            <p className="text-sm text-gray-500 mt-2">
              Supported formats: .EDF, .NPZ, .CSV, .PKL
            </p>
          </div>
          {file && (
            <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg flex items-center gap-3">
              <FileCheck className="w-6 h-6 text-green-600" />
              <div className="flex-1">
                <p className="font-semibold text-gray-900">{file.name}</p>
                <p className="text-sm text-gray-600">
                  Size: {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
          )}
          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-700">{error}</p>
            </div>
          )}
          <button
            onClick={handleAnalysis}
            disabled={!file || isAnalyzing}
            className="w-full mt-6 px-6 py-4 bg-linear-to-r from-red-600 to-red-700 text-white font-bold text-lg rounded-xl hover:shadow-lg transition transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center justify-center gap-2"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-6 h-6 animate-spin" />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <Activity className="w-6 h-6" />
                <span>Analyze for Seizures</span>
              </>
            )}
          </button>
        </div>
        {/* Results Section */}
        {result && (
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Analysis Results</h2>
            <div className="space-y-6">
              {/* Prediction */}
              <div className="p-6 bg-linear-to-r from-blue-50 to-purple-50 rounded-xl border border-blue-200">
                <p className="text-sm text-gray-600 mb-2">Prediction</p>
                <p className="text-3xl font-bold text-gray-900">{result.prediction}</p>
              </div>
              {/* Confidence */}
              {result.confidence && (
                <div className="p-6 bg-gray-50 rounded-xl">
                  <p className="text-sm text-gray-600 mb-2">Confidence Score</p>
                  <div className="flex items-center gap-4">
                    <div className="flex-1 bg-gray-200 rounded-full h-4">
                      <div
                        className="bg-linear-to-r from-green-500 to-green-600 h-4 rounded-full transition-all"
                        style={{ width: `${result.confidence}%` }}
                      ></div>
                    </div>
                    <span className="text-2xl font-bold text-gray-900">{result.confidence}%</span>
                  </div>
                </div>
              )}
              {/* Download Report */}
              {result.reportUrl && (
                <a
                  href={result.reportUrl}
                  download
                  className="flex items-center justify-center gap-2 w-full px-6 py-3 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition"
                >
                  <Download className="w-5 h-5" />
                  <span>Download PDF Report</span>
                </a>
              )}
              {/* Additional Info */}
              {result.message && (
                <div className="p-4 bg-blue-50 rounded-lg">
                  <p className="text-gray-700">{result.message}</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
