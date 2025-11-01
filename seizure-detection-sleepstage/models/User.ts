import mongoose, { Schema, Document } from "mongoose";
export interface IUser extends Document {
  name: string;
  email: string;
  password: string;
  resetPasswordToken?: string;
  resetPasswordExpires?: Date;
  createdAt: Date;
  lastLogin?: Date;
}
const UserSchema = new Schema<IUser>({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true, lowercase: true, trim: true },
  password: { type: String, required: true },
  resetPasswordToken: { type: String },
  resetPasswordExpires: { type: Date },
  createdAt: { type: Date, default: Date.now },
  lastLogin: { type: Date }
});
export default mongoose.models.User || mongoose.model<IUser>("User", UserSchema);
