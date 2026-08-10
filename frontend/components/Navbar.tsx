"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Cpu, Terminal, Layers, BarChart3, Play, User } from "lucide-react";
import ThemeToggle from "@/components/ThemeToggle";

function GithubIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
      <path d="M9 18c-4.51 2-5-2-7-2" />
    </svg>
  );
}

export default function Navbar() {
  const pathname = usePathname();

  const navItems = [
    { name: "Home", path: "/", icon: Cpu },
    { name: "Journey", path: "/journey", icon: Terminal },
    { name: "Architecture", path: "/architecture", icon: Layers },
    { name: "Results", path: "/results", icon: BarChart3 },
    { name: "Live Demo", path: "/demo", icon: Play },
    { name: "About", path: "/about", icon: User },
  ];

  return (
    <header className="sticky top-0 z-50 backdrop-blur-md bg-slate-950/80 border-b border-slate-800/80 transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Branding */}
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-500 p-0.5 shadow-lg shadow-indigo-500/20 group-hover:shadow-indigo-500/40 transition-all">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <Cpu className="w-5 h-5 text-indigo-400 group-hover:scale-110 transition-transform" />
              </div>
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                NanoLLM
              </span>
              <span className="text-[10px] font-mono text-indigo-400 -mt-1 tracking-wider uppercase">
                From Scratch GPT
              </span>
            </div>
          </Link>

          {/* Desktop Navigation Links */}
          <nav className="hidden md:flex items-center space-x-1 bg-slate-900/60 p-1.5 rounded-full border border-slate-800/60">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.path;

              return (
                <Link
                  key={item.path}
                  href={item.path}
                  className={`flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium transition-all ${
                    isActive
                      ? "bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-md shadow-indigo-500/25"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>

          {/* Theme Toggle & GitHub CTA Button */}
          <div className="flex items-center gap-3">
            <ThemeToggle />
            <a
              href="https://github.com/MuhammadAbdullah12-ux/NanoLLM"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-slate-900 dark:bg-slate-900 hover:bg-slate-800 border border-slate-700/80 text-slate-200 transition-all hover:border-slate-600 shadow-sm"
            >
              <GithubIcon className="w-4 h-4 text-indigo-400" />
              <span className="hidden sm:inline">GitHub Repo</span>
            </a>
          </div>

        </div>
      </div>
    </header>
  );
}
