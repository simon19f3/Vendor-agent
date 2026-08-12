import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Vendor-Assessment Agent",
  description: "Autonomous ReAct vendor-assessment agent demo",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div style={{ maxWidth: 960, margin: "0 auto", padding: "24px 16px" }}>
          <h1 style={{ marginBottom: 4 }}>Vendor-Assessment Agent</h1>
          <p style={{ color: "#666", marginTop: 0 }}>
            Submit a vendor request and watch the ReAct agent reason, call tools, and decide.
          </p>
          {children}
        </div>
      </body>
    </html>
  );
}
