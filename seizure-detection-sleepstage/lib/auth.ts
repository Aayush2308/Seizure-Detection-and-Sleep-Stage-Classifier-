import { NextResponse } from "next/server";
import jwt from "jsonwebtoken";
import { connectDB } from "./dbConnect";
import Session from "@/models/Session";
import User from "@/models/User";
export interface AuthRequest {
  user?: {
    id: string;
    email: string;
    name: string;
  };
}
export async function validateToken(token: string): Promise<AuthRequest | null> {
  try {
    if (!token) return null;
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as any;
    await connectDB();
    const session = await Session.findOne({ 
      token, 
      isValid: true,
      expiresAt: { $gt: new Date() }
    });
    if (!session) {
      return null;
    }
    const user = await User.findById(decoded.id).select("-password");
    if (!user) {
      return null;
    }
    return {
      user: {
        id: user._id.toString(),
        email: user.email,
        name: user.name
      }
    };
  } catch (error) {
    return null;
  }
}
export async function getAuthUser(req: Request): Promise<{ user: any } | NextResponse> {
  const authHeader = req.headers.get("authorization");
  const token = authHeader?.replace("Bearer ", "");
  if (!token) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const auth = await validateToken(token);
  if (!auth || !auth.user) {
    return NextResponse.json({ error: "Invalid or expired token" }, { status: 401 });
  }
  return { user: auth.user };
}
