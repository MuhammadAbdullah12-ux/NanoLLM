import Link from "next/link";
import { Cpu, Terminal, ArrowRight, CheckCircle2, Sparkles, Zap, ShieldCheck } from "lucide-react";

export default function Home() {
  return (
    <div className="space-y-16 py-6">
      
      {/* Hero Section */}
      <section className="text-center space-y-6 max-w-4xl mx-auto pt-8">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-indigo-950/80 border border-indigo-800/60 text-indigo-300 text-xs font-semibold shadow-inner">
          <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
          <span>4-Week, 26-Day From-Scratch GPT Build</span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white leading-tight">
          Building a <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">GPT Transformer</span> from Raw Math
        </h1>

        <p className="text-lg text-slate-300 max-w-2xl mx-auto leading-relaxed">
          No black boxes. No high-level wrappers. A complete, step-by-step engineering journey building an autoregressive Small Language Model from scratch in raw PyTorch, tokenizing with BPE, and fine-tuning on domain text.
        </p>

        {/* Hero CTA Buttons */}
        <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
          <Link
            href="/demo"
            className="flex items-center gap-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold text-sm hover:opacity-95 shadow-lg shadow-indigo-500/25 transition-all hover:scale-[1.02]"
          >
            <span>Try Live Model Demo</span>
            <ArrowRight className="w-4 h-4" />
          </Link>

          <Link
            href="/journey"
            className="flex items-center gap-2 px-6 py-3.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700/80 text-slate-200 font-semibold text-sm transition-all"
          >
            <Terminal className="w-4 h-4 text-indigo-400" />
            <span>Explore 26-Day Timeline</span>
          </Link>
        </div>
      </section>

      {/* Key Metric Highlights */}
      <section className="grid grid-cols-1 sm:grid-cols-3 gap-6 pt-6">
        <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 space-y-2 hover:border-slate-700 transition-all">
          <div className="text-xs font-mono text-slate-400 uppercase tracking-wider">Parameters</div>
          <div className="text-3xl font-extrabold text-white">211,777</div>
          <p className="text-xs text-slate-400">Custom Mini-GPT architecture with 4 layers, 4 attention heads, 64 embedding dim.</p>
        </div>

        <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 space-y-2 hover:border-slate-700 transition-all">
          <div className="text-xs font-mono text-slate-400 uppercase tracking-wider">Perplexity Reduction</div>
          <div className="text-3xl font-extrabold text-emerald-400">76.56 → 6.64</div>
          <p className="text-xs text-slate-400">Massive quality jump from untrained random initial weights to optimal convergence.</p>
        </div>

        <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 space-y-2 hover:border-slate-700 transition-all">
          <div className="text-xs font-mono text-slate-400 uppercase tracking-wider">BPE Compression</div>
          <div className="text-3xl font-extrabold text-purple-400">3.3× Compression</div>
          <p className="text-xs text-slate-400">Subword BPE tokenization using OpenAI tiktoken (|V| = 50,257).</p>
        </div>
      </section>

      {/* Architecture Highlights */}
      <section className="bg-slate-900/40 border border-slate-800/80 rounded-3xl p-8 sm:p-12 space-y-8">
        <div className="max-w-2xl space-y-3">
          <h2 className="text-2xl sm:text-3xl font-bold text-white">What Makes NanoLLM Unique?</h2>
          <p className="text-slate-400 text-sm">
            Built from scratch to demonstrate full mathematical understanding of modern generative AI foundation models.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="flex gap-4 p-5 rounded-xl bg-slate-900/80 border border-slate-800">
            <div className="p-3 rounded-lg bg-indigo-950 text-indigo-400 h-fit">
              <Zap className="w-5 h-5" />
            </div>
            <div className="space-y-1">
              <h3 className="font-semibold text-white text-sm">Causal Multi-Head Self-Attention</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Full PyTorch implementation of Query, Key, Value projections ($Q, K, V$), scaled dot-product attention, and triangular causal masking (`tril`).
              </p>
            </div>
          </div>

          <div className="flex gap-4 p-5 rounded-xl bg-slate-900/80 border border-slate-800">
            <div className="p-3 rounded-lg bg-purple-950 text-purple-400 h-fit">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div className="space-y-1">
              <h3 className="font-semibold text-white text-sm">Modern Optimization & Regularization</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Cosine Learning Rate Annealing with Warmup, Gradient Clipping ($1.0$), AdamW Weight Decay ($0.1$), and Inverted Dropout ($0.1$).
              </p>
            </div>
          </div>
        </div>
      </section>

    </div>
  );
}
