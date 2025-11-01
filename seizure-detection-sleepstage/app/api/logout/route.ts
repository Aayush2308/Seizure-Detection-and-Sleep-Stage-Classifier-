import { NextResponse } from "next/server";
import { connectDB } from "@/lib/dbConnect";
import Session from "@/models/Session";
export async function POST(req: Request) {
  try {
    const { token } = await req.json();
    if (!token) {
      return NextResponse.json({ error: "No token provided" }, { status: 400 });
    }
    await connectDB();
    await Session.updateOne(
      { token },
      { $set: { isValid: false } }
    );
    return NextResponse.json({ message: "Logged out successfully" }, { status: 200 });
  } catch (error) {
    console.error("Logout error:", error);
    return NextResponse.json({ error: "Something went wrong" }, { status: 500 });
  }
}
