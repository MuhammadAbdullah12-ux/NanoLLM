import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "NanoLLM Showcase — Building a GPT Transformer from Raw Math",
  description: "A 4-week, 26-day educational journey building, evaluating, and fine-tuning an autoregressive Small Language Model from scratch in raw PyTorch.",
  keywords: ["LLM", "GPT", "PyTorch", "Transformer", "Machine Learning", "Artificial Intelligence", "Fine-Tuning", "Subword BPE"],
  authors: [{ name: "Muhammad Abdullah" }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable} dark`}>
      <body className="bg-slate-950 text-slate-100 font-sans min-h-screen flex flex-col selection:bg-indigo-500 selection:text-white">
        <Navbar />
        <main className="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
