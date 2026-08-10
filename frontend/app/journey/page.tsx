import Timeline from "@/components/Timeline";
import { Terminal, Sparkles } from "lucide-react";

export default function JourneyPage() {
  return (
    <div className="space-y-8 py-6">
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-indigo-100 dark:bg-indigo-950/80 border border-indigo-300 dark:border-indigo-800/60 text-indigo-700 dark:text-indigo-400 text-xs font-semibold">
          <Terminal className="w-3.5 h-3.5" />
          <span>26-Day Interactive Build Timeline</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 dark:text-white">Engineering Journey</h1>
        <p className="text-slate-600 dark:text-slate-400 text-sm max-w-2xl">
          An interactive, week-by-week engineering breakdown of how NanoLLM evolved from simple PyTorch tensor math to a full GPT Transformer and fine-tuned foundation model. Click any week to reveal what broke and how we fixed it!
        </p>
      </div>

      <div className="pt-2">
        <Timeline />
      </div>
    </div>
  );
}
