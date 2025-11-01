import { NextResponse } from "next/server";
import { writeFile, mkdir, unlink } from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import { exec } from "child_process";
import { promisify } from "util";
import { connectDB } from "@/lib/dbConnect";
import { getAuthUser } from "@/lib/auth";
import Analysis from "@/models/Analysis";
const execAsync = promisify(exec);
export async function POST(req: Request) {
  let tempFilePath: string | null = null;
  try {
    const authResult = await getAuthUser(req);
    if (authResult instanceof NextResponse) {
      return authResult; 
    }
    const { user } = authResult;
    const formData = await req.formData();
    const file = formData.get("file") as File;
    const analysisType = formData.get("analysisType") as string;
    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }
    if (!analysisType || !["seizure", "sleep"].includes(analysisType)) {
      return NextResponse.json({ error: "Invalid analysis type" }, { status: 400 });
    }
    const uploadsDir = path.join(process.cwd(), "uploads");
    if (!existsSync(uploadsDir)) {
      await mkdir(uploadsDir, { recursive: true });
    }
    const reportsDir = path.join(process.cwd(), "public", "reports");
    if (!existsSync(reportsDir)) {
      await mkdir(reportsDir, { recursive: true });
    }
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const timestamp = Date.now();
    const uniqueFileName = `${user.id}_${timestamp}_${file.name}`;
    const filePath = path.join(uploadsDir, uniqueFileName);
    tempFilePath = filePath;
    await writeFile(filePath, buffer);
    const fileExtension = path.extname(file.name).toLowerCase().slice(1);
    const pythonScript = path.join(process.cwd(), "ml", "api_predict_enhanced.py");
    try {
      const reportName = `report_${user.id}_${timestamp}.pdf`;
      const reportPath = path.join(reportsDir, reportName);
      const { stdout, stderr } = await execAsync(
        `python "${pythonScript}" "${filePath}" --format ${fileExtension} --report "${reportPath}" --type ${analysisType}`,
        { maxBuffer: 1024 * 1024 * 10 } 
      );
      if (stderr && !stderr.includes("Warning") && !stderr.includes("FutureWarning")) {
        console.error("Python stderr:", stderr);
      }
      let result;
      try {
        const jsonMatch = stdout.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          result = JSON.parse(jsonMatch[0]);
        } else {
          throw new Error("No JSON found in output");
        }
      } catch (parseError) {
        console.error("Failed to parse Python output:", stdout);
        result = {
          prediction: extractPrediction(stdout),
          message: "Analysis completed but output format unexpected",
          rawOutput: stdout.substring(0, 500)
        };
      }
      if (result.error) {
        return NextResponse.json({
          error: result.error,
          suggestion: result.suggestion || "Please check the file format and try again."
        }, { status: 500 });
      }
      await connectDB();
      const analysis = await Analysis.create({
        userId: user.id,
        analysisType,
        fileName: file.name,
        fileFormat: fileExtension,
        prediction: result.prediction,
        predictionValue: result.prediction_value || 0,
        confidence: result.confidence,
        probabilities: result.probabilities,
        featuresExtracted: result.features_extracted,
        reportPath: `/reports/${reportName}`,
        metadata: {
          fileSize: file.size,
          originalName: file.name
        }
      });
      if (tempFilePath) {
        try {
          await unlink(tempFilePath);
        } catch (err) {
          console.error("Error deleting temp file:", err);
        }
      }
      return NextResponse.json({
        success: true,
        ...result,
        analysisId: analysis._id,
        reportUrl: `/reports/${reportName}`,
        analysisType
      });
    } catch (execError: any) {
      console.error("Python execution error:", execError);
      if (tempFilePath) {
        try {
          await unlink(tempFilePath);
        } catch (err) {
          console.error("Error deleting temp file:", err);
        }
      }
      return NextResponse.json({
        error: "Analysis failed",
        details: execError.message,
        suggestion: "Please ensure Python and required packages (numpy, pandas, scikit-learn, mne, scipy, matplotlib, reportlab) are installed."
      }, { status: 500 });
    }
  } catch (error: any) {
    console.error("API Error:", error);
    if (tempFilePath) {
      try {
        await unlink(tempFilePath);
      } catch (err) {
        console.error("Error deleting temp file:", err);
      }
    }
    return NextResponse.json({
      error: "Server error",
      details: error.message
    }, { status: 500 });
  }
}
function extractPrediction(output: string): string {
  if (output.toLowerCase().includes("seizure")) {
    return "Seizure Detected";
  } else if (output.toLowerCase().includes("normal") || output.toLowerCase().includes("no seizure")) {
    return "Normal (No Seizure)";
  } else if (output.toLowerCase().includes("wake")) {
    return "Wake Stage";
  } else if (output.toLowerCase().includes("rem")) {
    return "REM Sleep";
  } else if (output.toLowerCase().includes("nrem") || output.toLowerCase().includes("n1") || output.toLowerCase().includes("n2") || output.toLowerCase().includes("n3")) {
    return "NREM Sleep";
  }
  return "Analysis Complete";
}
export const config = {
  api: {
    bodyParser: false,
  },
};
