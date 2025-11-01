import { NextResponse } from "next/server";
import { getAuthUser } from "@/lib/auth";
export async function GET(req: Request) {
  const authResult = await getAuthUser(req);
  if (authResult instanceof NextResponse) {
    return authResult; 
  }
  return NextResponse.json({
    user: authResult.user
  });
}
