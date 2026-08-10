import ArchitectureDiagram from "@/components/ArchitectureDiagram";
import { Layers, Cpu, Zap, ShieldCheck } from "lucide-react";

export default function ArchitecturePage() {
  return (
    <div className="space-y-8 py-6">
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-purple-100 dark:bg-purple-950/80 border border-purple-300 dark:border-purple-800/60 text-purple-700 dark:text-purple-400 text-xs font-semibold">
          <Layers className="w-3.5 h-3.5" />
          <span>Interactive Computational Graph</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 dark:text-white">Transformer Architecture</h1>
        <p className="text-slate-600 dark:text-slate-400 text-sm max-w-2xl">
          An interactive, visual breakdown of the mathematical components comprising the NanoLLM autoregressive Transformer Block. Click any stage below to inspect tensor shapes, mathematical operators, and raw PyTorch source code!
        </p>
      </div>

      <div className="pt-2">
        <ArchitectureDiagram />
      </div>
    </div>
  );
}
