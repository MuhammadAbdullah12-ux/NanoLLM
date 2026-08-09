import { BarChart3, TrendingDown, Award } from "lucide-react";

export default function ResultsPage() {
  const benchmarks = [
    { model: "Un-Trained Random Initialized", params: "211,777", vocab: "65", loss: "4.3381", ppl: "76.56" },
    { model: "NanoLLM Mini-GPT (From Scratch)", params: "211,777", vocab: "65", loss: "1.8942", ppl: "6.64" },
    { model: "Pretrained GPT-2 Small (Hugging Face)", params: "124,439,808", vocab: "50,257", loss: "3.4210", ppl: "30.60" },
    { model: "Fine-Tuned GPT-2 Small (100 Steps)", params: "124,439,808", vocab: "50,257", loss: "2.1580", ppl: "8.65" },
  ];

  return (
    <div className="space-y-8 py-6">
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/80 border border-emerald-800/60 text-emerald-400 text-xs font-semibold">
          <BarChart3 className="w-3.5 h-3.5" />
          <span>Quantitative Benchmarks</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white">Experimental Results</h1>
        <p className="text-slate-400 text-sm max-w-2xl">
          Validation Loss & Perplexity ($PPL = e^L$) comparisons across random baseline, from-scratch Mini-GPT, and fine-tuned GPT-2 foundation models.
        </p>
      </div>

      <div className="overflow-x-auto bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-mono">
              <th className="pb-4 font-semibold">Model / Approach</th>
              <th className="pb-4 font-semibold">Parameters</th>
              <th className="pb-4 font-semibold">Vocab Size</th>
              <th className="pb-4 font-semibold">Val Loss</th>
              <th className="pb-4 font-semibold">Perplexity (PPL)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {benchmarks.map((row, idx) => (
              <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                <td className="py-4 font-semibold text-white">{row.model}</td>
                <td className="py-4 font-mono text-slate-300">{row.params}</td>
                <td className="py-4 font-mono text-indigo-400">{row.vocab}</td>
                <td className="py-4 font-mono text-slate-200">{row.loss}</td>
                <td className="py-4 font-mono font-bold text-emerald-400">{row.ppl}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
