"use client";

import { useState } from "react";
import { Calendar, CheckCircle2, AlertTriangle, Wrench, ChevronDown, ChevronUp, Code2 } from "lucide-react";

interface TimelineItem {
  week: string;
  days: string;
  title: string;
  summary: string;
  built: string[];
  broke: string;
  fix: string;
  codeSnippet: string;
}

export default function Timeline() {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(0);

  const timelineData: TimelineItem[] = [
    {
      week: "Week 1",
      days: "Days 1–7",
      title: "PyTorch Foundations & Diagnostic Experiments",
      summary: "Mastered PyTorch tensor operations, autograd backpropagation, and loss surface diagnostics.",
      built: [
        "4 Sacred Loop Steps (forward, loss, backward, optimizer step)",
        "MNIST PyTorch Classifier & loss landscape visualization",
        "Train/Val loss split monitoring to detect overfitting early"
      ],
      broke: "Setting learning rate too high (LR = 0.5) caused gradients to explode instantly into NaN loss.",
      fix: "Implemented Learning Rate bounds testing and established LR = 0.001 baseline with gradient diagnostics.",
      codeSnippet: "optimizer.zero_grad(set_to_none=True)\nloss.backward()\ntorch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\noptimizer.step()"
    },
    {
      week: "Week 2",
      days: "Days 8–14",
      title: "Building the GPT Transformer Architecture",
      summary: "Constructed complete autoregressive Transformer Block from scratch in raw PyTorch math.",
      built: [
        "Token Embedding Table & Learned Positional Encoding",
        "Causal Multi-Head Self-Attention with Q, K, V dot-product",
        "Feed-Forward Network (4x expansion) & Pre-LN Residual Blocks"
      ],
      broke: "Without causal masking, the model cheated by looking at future tokens during attention computation.",
      fix: "Built lower-triangular causal mask matrix (tril) filling upper-triangle with -infinity before Softmax.",
      codeSnippet: "wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)\nwei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))\nwei = F.softmax(wei, dim=-1)"
    },
    {
      week: "Week 3",
      days: "Days 15–21",
      title: "Optimization, Regularization & Advanced Sampling",
      summary: "Scaled architecture to 211,777 parameters and added production-grade sampling.",
      built: [
        "Cosine Learning Rate Scheduler with Warmup (WARMUP_ITERS=100)",
        "Inverted Dropout (0.1) & AdamW Decoupled Weight Decay (0.1)",
        "3-Tier Sampling Pipeline (Temperature T=0.8, Top-K K=20, Top-P P=0.9)"
      ],
      broke: "Pure greedy decoding (temperature = 0) resulted in repetitive infinite loops ('the king said the king said').",
      fix: "Combined Temperature scaling with Top-K truncation and Top-P nucleus sampling for natural text diversity.",
      codeSnippet: "logits = logits / temperature\nv, _ = torch.topk(logits, top_k)\nlogits[logits < v[:, [-1]]] = -float('Inf')\nprobs = F.softmax(logits, dim=-1)"
    },
    {
      week: "Week 4",
      days: "Days 22–26",
      title: "Subword BPE & Transfer Learning Fine-Tuning",
      summary: "Switched to OpenAI Subword BPE and fine-tuned 124M GPT-2 on domain text.",
      built: [
        "Subword Byte-Pair Encoding via tiktoken (50,257 vocab size)",
        "3.3x sequence compression factor & Int64 BPE tensor data loader",
        "Pretrained GPT-2 (124M) loading & 100-step fine-tuning on Shakespeare"
      ],
      broke: "Fine-tuning with a high learning rate (LR = 1e-3) destroyed pretrained weights (catastrophic forgetting).",
      fix: "Reduced fine-tuning learning rate to LR = 5e-5, preserving general language syntax while adapting domain style.",
      codeSnippet: "# Fine-tuning Hugging Face GPT-2 (124M)\noptimizer = optim.AdamW(model.parameters(), lr=5e-5)\noutputs = model(xb, labels=yb)\nloss = outputs.loss"
    }
  ];

  return (
    <div className="space-y-6">
      {timelineData.map((item, idx) => {
        const isExpanded = expandedIndex === idx;

        return (
          <div
            key={idx}
            className="bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 rounded-2xl p-6 transition-all shadow-sm"
          >
            {/* Card Header */}
            <div
              onClick={() => setExpandedIndex(isExpanded ? null : idx)}
              className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 cursor-pointer select-none"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-3">
                  <span className="px-3 py-1 rounded-md bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 text-xs font-bold font-mono">
                    {item.week} ({item.days})
                  </span>
                  <span className="text-xs text-slate-500 font-medium">Click to expand details</span>
                </div>
                <h3 className="text-xl font-bold text-slate-900 dark:text-white pt-1">{item.title}</h3>
              </div>

              <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 text-xs font-semibold">
                <span>{isExpanded ? "Collapse" : "View Breakdown"}</span>
                {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </div>
            </div>

            <p className="text-sm text-slate-600 dark:text-slate-300 pt-3 leading-relaxed">
              {item.summary}
            </p>

            {/* Expanded Content */}
            {isExpanded && (
              <div className="mt-6 pt-6 border-t border-slate-200 dark:border-slate-800/80 space-y-6 animate-fadeIn">
                
                {/* What We Built */}
                <div className="space-y-2">
                  <h4 className="text-xs font-mono uppercase tracking-wider text-slate-900 dark:text-slate-200 font-bold flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                    <span>What We Built</span>
                  </h4>
                  <ul className="grid grid-cols-1 md:grid-cols-3 gap-2 pt-1">
                    {item.built.map((b, bIdx) => (
                      <li key={bIdx} className="text-xs text-slate-700 dark:text-slate-300 bg-slate-100 dark:bg-slate-950 p-2.5 rounded-lg border border-slate-200 dark:border-slate-800">
                        {b}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* What Broke vs The Fix */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900/60 space-y-1">
                    <div className="flex items-center gap-2 text-xs font-bold text-amber-800 dark:text-amber-400">
                      <AlertTriangle className="w-4 h-4" />
                      <span>What Broke</span>
                    </div>
                    <p className="text-xs text-amber-900 dark:text-amber-200 leading-relaxed">{item.broke}</p>
                  </div>

                  <div className="p-4 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-900/60 space-y-1">
                    <div className="flex items-center gap-2 text-xs font-bold text-emerald-800 dark:text-emerald-400">
                      <Wrench className="w-4 h-4" />
                      <span>The Engineering Fix</span>
                    </div>
                    <p className="text-xs text-emerald-900 dark:text-emerald-200 leading-relaxed">{item.fix}</p>
                  </div>
                </div>

                {/* Key Code Snippet */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-xs font-mono font-bold text-slate-900 dark:text-slate-200">
                    <Code2 className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                    <span>Key Architectural Code / Formula</span>
                  </div>
                  <pre className="p-4 rounded-xl bg-slate-900 dark:bg-slate-950 text-indigo-300 dark:text-indigo-300 font-mono text-xs overflow-x-auto border border-slate-700 dark:border-slate-800">
                    <code>{item.codeSnippet}</code>
                  </pre>
                </div>

              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
