import { Terminal, Calendar, CheckCircle2 } from "lucide-react";

export default function JourneyPage() {
  const weeks = [
    {
      week: "Week 1",
      title: "PyTorch Foundations & Diagnostic Experiments",
      days: "Days 1–7",
      description: "Mastered PyTorch Tensors, Autograd, 4 Sacred Loop Steps, MNIST Classifier, Train/Val Loss Monitoring, Overfitting & Exploding Learning Rate Experiments.",
      highlights: ["Built PyTorch training loop from scratch", "Diagnosed exploding gradients & learning rate limits", "Established train/val split monitoring"]
    },
    {
      week: "Week 2",
      title: "Building the GPT Transformer Architecture",
      days: "Days 8–14",
      description: "Implemented Tokenization, Sequence Target Shifting ($y=x+1$), Token & Positional Embeddings, Causal Self-Attention, Multi-Head Attention, FFN, Pre-LN Residual Blocks, and Autoregressive Generation.",
      highlights: ["Implemented Q, K, V dot-product attention", "Built causal triangular mask matrix (tril)", "Assembled full Transformer Block & LM Head"]
    },
    {
      week: "Week 3",
      title: "Optimization, Regularization & Advanced Sampling",
      days: "Days 15–21",
      description: "Integrated Cosine LR Scheduler with Warmup, Gradient Clipping, AdamW Weight Decay, Architecture Scaling (211k params), Inverted Dropout, Temperature Scaling, Top-K, Top-P Nucleus Sampling, and Perplexity Benchmarking.",
      highlights: ["Scaled architecture to 211,777 parameters", "Implemented Temperature + Top-K + Top-P 3-tier sampling pipeline", "Achieved Validation Loss 1.8942 / Perplexity 6.64"]
    },
    {
      week: "Week 4",
      title: "Subword BPE & Transfer Learning Fine-Tuning",
      days: "Days 22–26",
      description: "Integrated Subword Byte-Pair Encoding (`tiktoken` `gpt2`) with 3.3× compression factor, adapted vocabulary to 50,257 tokens, loaded pretrained GPT-2 Small (124M) from Hugging Face, and fine-tuned on Shakespeare dialogue.",
      highlights: ["Proved 3.3x subword sequence compression factor", "Loaded 124M pretrained GPT-2 from Hugging Face", "Fine-tuned 124M model on Shakespeare dialogue for 100 steps"]
    }
  ];

  return (
    <div className="space-y-8 py-6">
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-950/80 border border-indigo-800/60 text-indigo-400 text-xs font-semibold">
          <Terminal className="w-3.5 h-3.5" />
          <span>26-Day Build Timeline</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white">Engineering Journey</h1>
        <p className="text-slate-400 text-sm max-w-2xl">
          A week-by-week timeline of how NanoLLM evolved from simple PyTorch tensor math to a full GPT Transformer and fine-tuned foundation model.
        </p>
      </div>

      <div className="space-y-6 pt-4">
        {weeks.map((item, idx) => (
          <div key={idx} className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 space-y-4 hover:border-indigo-900/50 transition-all">
            <div className="flex items-center justify-between">
              <span className="px-3 py-1 rounded-md bg-indigo-950 text-indigo-300 text-xs font-bold font-mono">
                {item.week} ({item.days})
              </span>
              <Calendar className="w-4 h-4 text-slate-500" />
            </div>
            <h2 className="text-xl font-bold text-white">{item.title}</h2>
            <p className="text-sm text-slate-300 leading-relaxed">{item.description}</p>
            
            <div className="pt-2 border-t border-slate-800/60 grid grid-cols-1 sm:grid-cols-3 gap-3">
              {item.highlights.map((h, hIdx) => (
                <div key={hIdx} className="flex items-center gap-2 text-xs text-slate-400">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                  <span>{h}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
