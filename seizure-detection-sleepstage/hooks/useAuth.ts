"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
export interface User {
  id: string;
  name: string;
  email: string;
}
export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  useEffect(() => {
    const fetchUser = async () => {
      const token = localStorage.getItem("token");
      if (!token) {
        router.push("/login");
        setLoading(false);
        return;
      }
      try {
        const res = await fetch("/api/user/me", {
          headers: {
            "Authorization": `Bearer ${token}`
          }
        });
        if (res.ok) {
          const data = await res.json();
          setUser(data.user);
        } else {
          localStorage.removeItem("token");
          router.push("/login");
        }
      } catch (error) {
        console.error("Failed to fetch user:", error);
        router.push("/login");
      } finally {
        setLoading(false);
      }
    };
    fetchUser();
  }, [router]);
  return { user, loading };
}
