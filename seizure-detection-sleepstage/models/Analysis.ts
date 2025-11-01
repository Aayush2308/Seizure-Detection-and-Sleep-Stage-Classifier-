import mongoose, { Schema, Document } from "mongoose";
export interface IAnalysis extends Document {
  userId: mongoose.Types.ObjectId;
  analysisType: "seizure" | "sleep";
  fileName: string;
  fileFormat: string;
  prediction: string;
  predictionValue: number;
  confidence: number;
  probabilities?: {
    normal?: number;
    seizure?: number;
    wake?: number;
    rem?: number;
    nrem?: number;
  };
  featuresExtracted?: number;
  reportPath?: string;
  createdAt: Date;
  metadata?: any;
}
const AnalysisSchema = new Schema<IAnalysis>({
  userId: { 
    type: Schema.Types.ObjectId, 
    ref: "User", 
    required: true,
    index: true 
  },
  analysisType: { 
    type: String, 
    enum: ["seizure", "sleep"], 
    required: true 
  },
  fileName: { type: String, required: true },
  fileFormat: { type: String, required: true },
  prediction: { type: String, required: true },
  predictionValue: { type: Number, required: true },
  confidence: { type: Number },
  probabilities: {
    normal: Number,
    seizure: Number,
    wake: Number,
    rem: Number,
    nrem: Number,
  },
  featuresExtracted: { type: Number },
  reportPath: { type: String },
  createdAt: { type: Date, default: Date.now },
  metadata: { type: Schema.Types.Mixed }
});
export default mongoose.models.Analysis || mongoose.model<IAnalysis>("Analysis", AnalysisSchema);
