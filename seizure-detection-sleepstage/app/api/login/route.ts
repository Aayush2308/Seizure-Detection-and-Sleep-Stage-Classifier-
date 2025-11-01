import { NextResponse } from "next/server";
import { connectDB } from "@/lib/dbConnect";
import User from "@/models/User";
import Session from "@/models/Session";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";

export async function POST(req: Request) {
  try {
    const { email, password } = await req.json();
    await connectDB();

    const user = await User.findOne({ email });
    if (!user) {
      return NextResponse.json({ message: "Invalid email or password" }, { status: 400 });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return NextResponse.json({ message: "Invalid email or password" }, { status: 400 });
    }

    const token = jwt.sign(
      { id: user._id, email: user.email },
      process.env.JWT_SECRET!,
      { expiresIn: "7d" }
    );

    // Create session in database
    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + 7); // 7 days

    await Session.create({
      userId: user._id,
      token,
      expiresAt,
      isValid: true
    });

    // Update last login
    user.lastLogin = new Date();
    await user.save();

    return NextResponse.json({ 
      message: "Login successful", 
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email
      }
    }, { status: 200 });
  } catch (error) {
    console.error("Login error:", error);
    return NextResponse.json({ error: "Something went wrong" }, { status: 500 });
  }
}
