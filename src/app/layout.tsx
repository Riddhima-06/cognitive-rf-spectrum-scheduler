import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cognitive ES Platform | Automated Blind RF Parameter Extraction & Bandit Scheduler",
  description:
    "Interactive Cognitive Electronic Support (ES) simulation platform featuring real-time Hann-STFT, 2D CA-CFAR blind parameter extraction, 1D-CNN modulation recognition, and Non-Stationary Multi-Armed Bandit (UCB1 / Thompson Sampling) spectrum scheduling.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#060911] text-slate-100 antialiased selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
