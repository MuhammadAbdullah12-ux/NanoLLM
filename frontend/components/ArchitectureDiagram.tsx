"use client";

import { useState } from "react";
import { Cpu, ArrowDown, Layers, Zap, ShieldCheck, Code2, Sparkles, CheckCircle2 } from "lucide-react";

interface NodeDetail {
  id: string;
  title: string;
  subtitle: string;
  tensorShape: string;
  formula: string;
  codeSnippet: string;
  explanation: string;
}

export default function ArchitectureDiagram() {
  const [selectedNode, setSelectedNode] = useState<string>("attention");

  const nodes: Record<string, NodeDetail> = {
    embedding: {
      id: "embedding",
      title: "1. Token & Positional Embeddings",
      subtitle: "Maps Discrete Token IDs into Continuous Vector Space + Sequence Position",
      tensorShape: "[Batch Size, Sequence Length, Embedding Dim]  ->  [B, T, C]",
      formula: "x = W_E[idx] + W_P[pos]",
      codeSnippet: "tok_emb = self.token_embedding_table(idx) # [B, T, 64]\npos_emb = self.position_embedding_table(pos) # [T, 64]\nx = tok_emb + pos_emb # Additive Positional Encoding",
      explanation: "Converts discrete token numbers into dense 64-dimensional vectors while adding learned positional embedding vectors so the model knows spatial word order."
    },
    attention: {
      id: "attention",
      title: "2. Causal Multi-Head Self-Attention",
      subtitle: "Scaled Dot-Product Attention across 4 Parallel Heads with Causal Masking",
      tensorShape: "Q, K, V Shapes: [B, H, T, Head Size]  ->  [16, 4, 64, 16]",
      formula: "Attention(Q, K, V) = Softmax( (Q K^T / sqrt(d_k)) + TrilMask ) V",
      codeSnippet: "# Query, Key, Value Projections\nq = self.q_proj(x).view(B, T, n_head, head_size).transpose(1, 2)\nk = self.k_proj(x).view(B, T, n_head, head_size).transpose(1, 2)\nv = self.v_proj(x).view(B, T, n_head, head_size).transpose(1, 2)\n\n# Scaled Dot-Product + Causal Masking\nwei = q @ k.transpose(-2, -1) * (head_size ** -0.5)\nwei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))\nout = F.softmax(wei, dim=-1) @ v",
      explanation: "Allows tokens to attend to previous tokens in the sequence. Calculates Query-Key similarity scores scaled by 1/sqrt(d_k), masks out future tokens with lower-triangular matrix, and aggregates Values."
    },
    ffn: {
      id: "ffn",
      title: "3. Feed-Forward Neural Network (FFN)",
      subtitle: "4x Hidden Dimension Expansion with Non-Linear GELU Activation",
      tensorShape: "[B, T, C]  ->  [B, T, 4C]  ->  [B, T, C]",
      formula: "FFN(x) = GELU(x W_1 + b_1) W_2 + b_2",
      codeSnippet: "self.ffwd = nn.Sequential(\n    nn.Linear(config.N_EMBD, 4 * config.N_EMBD), # [64 -> 256]\n    nn.GELU(),\n    nn.Linear(4 * config.N_EMBD, config.N_EMBD), # [256 -> 64]\n    nn.Dropout(config.DROPOUT)\n)",
      explanation: "Applies a 4x expansion linear transformation followed by non-linear GELU activation. Acts as a position-wise associative memory for factual knowledge."
    },
    residuals: {
      id: "residuals",
      title: "4. Pre-LN Residual Skip Connections",
      subtitle: "LayerNorm before Sub-Layers + Additive Skip Connections",
      tensorShape: "[B, T, C] (Shape Preserved Across All Blocks)",
      formula: "x_{next} = x + Attention(LN(x)) + FFN(LN(x_attn))",
      codeSnippet: "# Pre-LayerNorm Transformer Block Forward Pass\nx = x + self.sa(self.ln1(x))   # Residual Skip Connection 1\nx = x + self.ffwd(self.ln2(x)) # Residual Skip Connection 2",
      explanation: "Applies Layer Normalization before attention and feed-forward sub-layers (Pre-LN). Additive residual skip connections allow gradients to flow straight through deep networks without vanishing."
    },
    head: {
      id: "head",
      title: "5. Final LayerNorm & Un-normalized LM Head",
      subtitle: "Projects 64-Dim Hidden State to Target Vocabulary Probability Logits",
      tensorShape: "[B, T, N_EMBD]  ->  [B, T, Vocab Size]",
      formula: "z = LN(x) W_{head}",
      codeSnippet: "x = self.ln_f(x) # Final Layer Normalization\nlogits = self.lm_head(x) # Linear Projection [64 -> 50,257]\nloss = F.cross_entropy(logits.view(-1, V), targets.view(-1))",
      explanation: "Normalizes final block hidden states and projects 64-dimensional vectors to vocabulary size channels (50,257 for BPE), computing Cross-Entropy Loss against targets."
    }
  };

  const selected = nodes[selectedNode];

  return (
    <div className="space-y-8">
      
      {/* Interactive Visual Graph Nodes */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
        {Object.values(nodes).map((node) => {
          const isActive = selectedNode === node.id;
          return (
            <button
              key={node.id}
              onClick={() => setSelectedNode(node.id)}
              className={`p-4 rounded-xl text-left border transition-all flex flex-col justify-between ${
                isActive
                  ? "bg-gradient-to-br from-indigo-600 to-purple-600 text-white border-indigo-500 shadow-lg shadow-indigo-500/25 scale-[1.02]"
                  : "bg-white dark:bg-slate-900/60 text-slate-800 dark:text-slate-200 border-slate-300 dark:border-slate-800 hover:border-indigo-400"
              }`}
            >
              <div className="space-y-1">
                <span className="text-[10px] font-mono uppercase tracking-wider opacity-80">Stage</span>
                <h4 className="font-bold text-xs leading-snug">{node.title.split(". ")[1]}</h4>
              </div>
              <span className="text-[10px] font-mono mt-3 opacity-90 block">{node.tensorShape.split("->")[0]}</span>
            </button>
          );
        })}
      </div>

      {/* Selected Node Deep-Dive Panel */}
      <div className="bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 rounded-2xl p-6 sm:p-8 space-y-6 shadow-sm animate-fadeIn">
        
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-4">
          <div className="space-y-1">
            <span className="px-3 py-1 rounded-md bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 text-xs font-bold font-mono">
              {selected.id.toUpperCase()} MODULE
            </span>
            <h3 className="text-2xl font-extrabold text-slate-900 dark:text-white pt-1">{selected.title}</h3>
            <p className="text-xs text-slate-600 dark:text-slate-400">{selected.subtitle}</p>
          </div>

          <div className="px-4 py-2 rounded-xl bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 font-mono text-xs text-indigo-600 dark:text-indigo-400">
            Shape: {selected.tensorShape}
          </div>
        </div>

        <p className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed font-medium">
          {selected.explanation}
        </p>

        {/* Mathematical Equation */}
        <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-950/80 border border-slate-200 dark:border-slate-800 space-y-2">
          <div className="text-xs font-mono font-bold text-slate-900 dark:text-slate-200 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-purple-600 dark:text-purple-400" />
            <span>Mathematical Operator Equation</span>
          </div>
          <div className="font-mono text-sm text-purple-700 dark:text-purple-300 font-bold overflow-x-auto">
            <code>{selected.formula}</code>
          </div>
        </div>

        {/* PyTorch Source Code */}
        <div className="space-y-2">
          <div className="text-xs font-mono font-bold text-slate-900 dark:text-slate-200 flex items-center gap-2">
            <Code2 className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
            <span>Raw PyTorch Source Code</span>
          </div>
          <pre className="p-4 rounded-xl bg-slate-900 text-indigo-300 font-mono text-xs overflow-x-auto border border-slate-700 dark:border-slate-800">
            <code>{selected.codeSnippet}</code>
          </pre>
        </div>

      </div>

    </div>
  );
}
