import mongoose, { Schema, Document } from "mongoose";
export interface ISession extends Document {
  userId: mongoose.Types.ObjectId;
  token: string;
  createdAt: Date;
  expiresAt: Date;
  isValid: boolean;
}
const SessionSchema = new Schema<ISession>({
  userId: { 
    type: Schema.Types.ObjectId, 
    ref: "User", 
    required: true,
    index: true 
  },
  token: { type: String, required: true, unique: true },
  createdAt: { type: Date, default: Date.now },
  expiresAt: { type: Date, required: true },
  isValid: { type: Boolean, default: true }
});
SessionSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 });
export default mongoose.models.Session || mongoose.model<ISession>("Session", SessionSchema);
