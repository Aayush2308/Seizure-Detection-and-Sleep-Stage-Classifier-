import { NextResponse } from "next/server";
import { connectDB } from "@/lib/dbConnect";
import { getAuthUser } from "@/lib/auth";
import User from "@/models/User";
import Session from "@/models/Session";
import Analysis from "@/models/Analysis";
import bcrypt from "bcrypt";
import { unlink } from "fs/promises";
import path from "path";
export async function POST(req: Request) {
  try {
    const authResult = await getAuthUser(req);
    if (authResult instanceof NextResponse) {
      return authResult; 
    }
    const { user } = authResult;
    const { password } = await req.json();
    if (!password) {
      return NextResponse.json({ error: "Password is required to delete account" }, { status: 400 });
    }
    await connectDB();
    const userDoc = await User.findById(user.id);
    if (!userDoc) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }
    const isMatch = await bcrypt.compare(password, userDoc.password);
    if (!isMatch) {
      return NextResponse.json({ error: "Incorrect password" }, { status: 400 });
    }
    const analyses = await Analysis.find({ userId: user.id });
    for (const analysis of analyses) {
      if (analysis.reportPath) {
        try {
          const reportPath = path.join(process.cwd(), analysis.reportPath);
          await unlink(reportPath);
        } catch (err) {
          console.error("Error deleting report file:", err);
        }
      }
    }
    await Analysis.deleteMany({ userId: user.id });
    await Session.deleteMany({ userId: user.id });
    await User.findByIdAndDelete(user.id);
    return NextResponse.json({ 
      message: "Account deleted successfully" 
    }, { status: 200 });
  } catch (error) {
    console.error("Delete account error:", error);
    return NextResponse.json({ error: "Something went wrong" }, { status: 500 });
  }
}
