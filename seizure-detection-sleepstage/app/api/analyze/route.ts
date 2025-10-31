import { NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File;
    const analysisType = formData.get("analysisType") as string;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // Create uploads directory if it doesn't exist
    const uploadsDir = path.join(process.cwd(), "uploads");
    if (!existsSync(uploadsDir)) {
      await mkdir(uploadsDir, { recursive: true });
    }

    // Save the uploaded file
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const filePath = path.join(uploadsDir, file.name);
    await writeFile(filePath, buffer);

    // Get file extension
    const fileExtension = path.extname(file.name).toLowerCase().slice(1);

    // Path to the Python API wrapper script
    const pythonScript = path.join(process.cwd(), "ml", "api_predict.py");

    // Execute Python script
    try {
      const { stdout, stderr } = await execAsync(
        `python "${pythonScript}" "${filePath}" --format ${fileExtension}`,
        { maxBuffer: 1024 * 1024 * 10 } // 10MB buffer
      );

      if (stderr && !stderr.includes("Warning") && !stderr.includes("FutureWarning")) {
        console.error("Python stderr:", stderr);
      }

      // Parse the JSON output from Python script
      let result;
      try {
        // Find the JSON in the output (in case there are warnings before it)
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
          rawOutput: stdout.substring(0, 500) // Limit output size
        };
      }

      // Check if Python script reported an error
      if (result.error) {
        return NextResponse.json({
          error: result.error,
          suggestion: result.suggestion || "Please check the file format and try again."
        }, { status: 500 });
      }

      return NextResponse.json({
        success: true,
        ...result,
        analysisType
      });

    } catch (execError: any) {
      console.error("Python execution error:", execError);
      
      return NextResponse.json({
        error: "Analysis failed",
        details: execError.message,
        suggestion: "Please ensure Python and required packages (numpy, pandas, scikit-learn, mne, scipy) are installed."
      }, { status: 500 });
    }

  } catch (error: any) {
    console.error("API Error:", error);
    return NextResponse.json({
      error: "Server error",
      details: error.message
    }, { status: 500 });
  }
}

// Helper function to extract prediction from output
function extractPrediction(output: string): string {
  // Look for common patterns in the output
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
