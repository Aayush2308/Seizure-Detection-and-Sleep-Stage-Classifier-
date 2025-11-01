import { NextResponse } from "next/server";
import { connectDB } from "@/lib/dbConnect";
import User from "@/models/User";
import crypto from "crypto";
export async function POST(req: Request) {
  try {
    const { email } = await req.json();
    if (!email) {
      return NextResponse.json({ error: "Email is required" }, { status: 400 });
    }
    await connectDB();
    const user = await User.findOne({ email: email.toLowerCase() });
    if (!user) {
      return NextResponse.json({ 
        message: "If an account with that email exists, a password reset link has been sent." 
      }, { status: 200 });
    }
    const resetToken = crypto.randomBytes(32).toString("hex");
    const hashedToken = crypto.createHash("sha256").update(resetToken).digest("hex");
    user.resetPasswordToken = hashedToken;
    user.resetPasswordExpires = new Date(Date.now() + 3600000);
    await user.save();
    
    const resetUrl = `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/reset-password?token=${resetToken}`;
    console.log("Password reset URL:", resetUrl);
    
    return NextResponse.json({ 
      message: "If an account with that email exists, a password reset link has been sent.",
      resetUrl: process.env.NODE_ENV === 'development' ? resetUrl : undefined
    }, { status: 200 });
  } catch (error) {
    console.error("Forgot password error:", error);
    return NextResponse.json({ error: "Something went wrong" }, { status: 500 });
  }
}
