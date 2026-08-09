import { Layers, Cpu, Zap, Activity } from "lucide-react";

export default function ArchitecturePage() {
  const components = [
    {
      title: "1. Token & Positional Embeddings",
      code: "x = token_embedding(idx) + position_embedding(pos)",
      desc: "Combines semantic token ID lookup representations with learned absolute positional embeddings to encode sequence spatial order."
    },
    {
      title: "2. Causal Multi-Head Self-Attention",
      code: "A = softmax((Q @ K.T) / sqrt(d_k) + tril_mask) @ V",
      desc: "Projects hidden states into Query (Q), Key (K), and Value (V) heads across 4 parallel heads, applying triangular masking to enforce autoregressive causality."
    },
    {
      title: "3. Feed-Forward Network (FFN)",
      code: "ffn(x) = Linear(64 -> 256) -> GELU() -> Linear(256 -> 64)",
      desc: "Applies a 4x dimension expansion linear projection followed by non-linear GELU activation and projection back to embedding dimension."
    },
    {
      title: "4. Layer Normalization & Residual Skip Connections",
      code: "x = x + attention(LayerNorm(x))",
      desc: "Uses Pre-LN architecture where LayerNorm stabilizes input variance before sub-layers, combined with residual additive skip connections to prevent vanishing gradients."
    }
  ];

  return (
    <div className="space-y-8 py-6">
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-950/80 border border-indigo-800/60 text-indigo-400 text-xs font-semibold">
          <Layers className="w-3.5 h-3.5" />
          <span>Transformer Deep Dive</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white">NanoLLM Architecture</h1>
        <p className="text-slate-400 text-sm max-w-2xl">
          An interactive breakdown of the mathematical components comprising the NanoLLM autoregressive Transformer Block.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
        {components.map((comp, idx) => (
          <div key={idx} className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 space-y-4 hover:border-purple-900/50 transition-all">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Cpu className="w-4 h-4 text-indigo-400" />
              <span>{comp.title}</span>
            </h2>
            <div className="p-3 bg-slate-950 rounded-xl font-mono text-xs text-indigo-300 border border-slate-800">
              <code>{comp.code}</code>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">{comp.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
