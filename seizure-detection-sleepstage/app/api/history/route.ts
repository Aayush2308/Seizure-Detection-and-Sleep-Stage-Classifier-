import { NextResponse } from "next/server";
import { connectDB } from "@/lib/dbConnect";
import { getAuthUser } from "@/lib/auth";
import Analysis from "@/models/Analysis";
export async function GET(req: Request) {
  try {
    const authResult = await getAuthUser(req);
    if (authResult instanceof NextResponse) {
      return authResult;
    }
    const { user } = authResult;
    await connectDB();
    const analyses = await Analysis.find({ userId: user.id })
      .sort({ createdAt: -1 })
      .limit(50); 
    return NextResponse.json({
      success: true,
      analyses: analyses.map(analysis => ({
        id: analysis._id,
        analysisType: analysis.analysisType,
        fileName: analysis.fileName,
        fileFormat: analysis.fileFormat,
        prediction: analysis.prediction,
        confidence: analysis.confidence,
        reportUrl: analysis.reportPath,
        createdAt: analysis.createdAt
      }))
    }, { status: 200 });
  } catch (error) {
    console.error("History error:", error);
    return NextResponse.json({ error: "Something went wrong" }, { status: 500 });
  }
}
