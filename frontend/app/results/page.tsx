import ResultsChart from "@/components/ResultsChart";
import { BarChart3, Cpu, Sparkles, TrendingDown, Award } from "lucide-react";

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
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-100 dark:bg-emerald-950/80 border border-emerald-300 dark:border-emerald-800/60 text-emerald-700 dark:text-emerald-400 text-xs font-semibold">
          <BarChart3 className="w-3.5 h-3.5" />
          <span>Quantitative Benchmarks</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 dark:text-white">Experimental Results</h1>
        <p className="text-slate-600 dark:text-slate-400 text-sm max-w-2xl">
          Validation Loss and Perplexity ($PPL = e^L$) benchmarks comparing random baselines, from-scratch Mini-GPT, and fine-tuned GPT-2 foundation models.
        </p>
      </div>

      {/* Visual Bar Chart */}
      <div className="pt-2">
        <ResultsChart />
      </div>

      {/* Secondary Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 rounded-2xl p-6 space-y-2 shadow-sm">
          <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 text-xs font-mono font-bold uppercase">
            <Cpu className="w-4 h-4" />
            <span>Parameter Efficiency</span>
          </div>
          <div className="text-2xl font-extrabold text-slate-900 dark:text-white">211k vs 124M</div>
          <p className="text-xs text-slate-600 dark:text-slate-400">NanoLLM achieves 6.64 PPL with $587\times$ fewer parameters than GPT-2 Small.</p>
        </div>

        <div className="bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 rounded-2xl p-6 space-y-2 shadow-sm">
          <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 text-xs font-mono font-bold uppercase">
            <TrendingDown className="w-4 h-4" />
            <span>Loss Reduction</span>
          </div>
          <div className="text-2xl font-extrabold text-emerald-600 dark:text-emerald-400">4.33 → 1.89</div>
          <p className="text-xs text-slate-600 dark:text-slate-400">Cross-entropy loss reduced by 56% over 2,000 optimization steps.</p>
        </div>

        <div className="bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 rounded-2xl p-6 space-y-2 shadow-sm">
          <div className="flex items-center gap-2 text-purple-600 dark:text-purple-400 text-xs font-mono font-bold uppercase">
            <Sparkles className="w-4 h-4" />
            <span>BPE Token Compression</span>
          </div>
          <div className="text-2xl font-extrabold text-purple-600 dark:text-purple-400">3.3× Factor</div>
          <p className="text-xs text-slate-600 dark:text-slate-400">Subword tokenization shrinks sequence length by $70\%$, accelerating attention.</p>
        </div>
      </div>

      {/* Raw Data Table */}
      <div className="bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 rounded-2xl p-6 space-y-4 shadow-sm">
        <h3 className="text-lg font-bold text-slate-900 dark:text-white">Raw Quantitative Benchmark Data</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 font-mono">
                <th className="pb-3 font-semibold">Model / Approach</th>
                <th className="pb-3 font-semibold">Parameters</th>
                <th className="pb-3 font-semibold">Vocab Size</th>
                <th className="pb-3 font-semibold">Val Loss</th>
                <th className="pb-3 font-semibold">Perplexity (PPL)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
              {benchmarks.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-800/30 transition-colors">
                  <td className="py-3 font-semibold text-slate-900 dark:text-white">{row.model}</td>
                  <td className="py-3 font-mono text-slate-600 dark:text-slate-300">{row.params}</td>
                  <td className="py-3 font-mono text-indigo-600 dark:text-indigo-400">{row.vocab}</td>
                  <td className="py-3 font-mono text-slate-700 dark:text-slate-200">{row.loss}</td>
                  <td className="py-3 font-mono font-bold text-emerald-600 dark:text-emerald-400">{row.ppl}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
