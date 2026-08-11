import Link from "next/link";
import { Cpu, ExternalLink, Heart } from "lucide-react";

function GithubIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
      <path d="M9 18c-4.51 2-5-2-7-2" />
    </svg>
  );
}

export default function Footer() {
  return (
    <footer className="bg-slate-100 dark:bg-slate-950 border-t border-slate-300 dark:border-slate-800/80 text-slate-600 dark:text-slate-400 text-xs py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          
          {/* Col 1: Brand pitch */}
          <div className="space-y-3 md:col-span-2">
            <div className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
              <span className="font-bold text-base text-slate-900 dark:text-white">NanoLLM Showcase</span>
            </div>
            <p className="text-slate-600 dark:text-slate-400 leading-relaxed max-w-md">
              A 4-week, 26-day educational journey building an autoregressive Small Language Model (GPT Transformer architecture) from scratch in raw PyTorch and fine-tuning on subwords.
            </p>
            <div className="flex items-center gap-2 pt-2">
              <span className="px-2.5 py-1 rounded-md bg-indigo-100 dark:bg-indigo-950/60 border border-indigo-300 dark:border-indigo-800/50 text-indigo-700 dark:text-indigo-400 font-mono text-[10px]">
                211,777 Params
              </span>
              <span className="px-2.5 py-1 rounded-md bg-purple-100 dark:bg-purple-950/60 border border-purple-300 dark:border-purple-800/50 text-purple-700 dark:text-purple-400 font-mono text-[10px]">
                Perplexity 6.64
              </span>
              <span className="px-2.5 py-1 rounded-md bg-emerald-100 dark:bg-emerald-950/60 border border-emerald-300 dark:border-emerald-800/50 text-emerald-700 dark:text-emerald-400 font-mono text-[10px]">
                PyTorch + Next.js
              </span>
            </div>
          </div>

          {/* Col 2: Navigation Links */}
          <div className="space-y-3">
            <h3 className="font-semibold text-slate-900 dark:text-slate-200 uppercase tracking-wider text-[11px]">Pages</h3>
            <ul className="space-y-2 font-medium">
              <li><Link href="/" className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">Home & Overview</Link></li>
              <li><Link href="/journey" className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">Build Journey (Days 1–26)</Link></li>
              <li><Link href="/architecture" className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">Transformer Architecture</Link></li>
              <li><Link href="/results" className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">Benchmark Charts & Results</Link></li>
              <li><Link href="/demo" className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">Live Model Demo</Link></li>
              <li><Link href="/about" className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">About & Freelance Contact</Link></li>
            </ul>
          </div>

          {/* Col 3: Resources & Socials */}
          <div className="space-y-3">
            <h3 className="font-semibold text-slate-900 dark:text-slate-200 uppercase tracking-wider text-[11px]">Resources</h3>
            <ul className="space-y-2 font-medium">
              <li>
                <a
                  href="https://github.com/MuhammadAbdullah12-ux/NanoLLM"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                >
                  <GithubIcon className="w-3.5 h-3.5" />
                  <span>GitHub Repository</span>
                  <ExternalLink className="w-3 h-3 text-slate-500" />
                </a>
              </li>
              <li>
                <a
                  href="https://huggingface.co/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                >
                  <span>Hugging Face Hub</span>
                  <ExternalLink className="w-3 h-3 text-slate-500" />
                </a>
              </li>
            </ul>
          </div>

        </div>

        <div className="pt-8 border-t border-slate-300 dark:border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4 text-slate-600 dark:text-slate-400 font-medium text-xs">
          <p>© 2026 NanoLLM Showcase. Open-Source Machine Learning Project.</p>
          <p className="text-slate-700 dark:text-slate-300 font-semibold">
            Developed by Muhammad Abdullah
          </p>
        </div>
      </div>
    </footer>
  );
}
