"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from "recharts";

export default function ResultsChart() {
  const data = [
    {
      name: "Random Baseline",
      shortName: "Random",
      loss: 4.3381,
      ppl: 76.56,
      color: "#ef4444" // Red
    },
    {
      name: "NanoLLM (Scratch)",
      shortName: "NanoLLM",
      loss: 1.8942,
      ppl: 6.64,
      color: "#10b981" // Emerald
    },
    {
      name: "Pretrained GPT-2",
      shortName: "GPT-2 Base",
      loss: 3.421,
      ppl: 30.6,
      color: "#6366f1" // Indigo
    },
    {
      name: "Fine-Tuned GPT-2",
      shortName: "GPT-2 Fine-Tuned",
      loss: 2.158,
      ppl: 8.65,
      color: "#a855f7" // Purple
    }
  ];

  return (
    <div className="space-y-6">
      
      {/* Chart 1: Perplexity (PPL) Bar Chart */}
      <div className="bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 rounded-2xl p-6 space-y-4 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">Validation Perplexity ($PPL = e^L$)</h3>
            <p className="text-xs text-slate-500">Lower is better. Measures model surprise/confusion on unseen validation text.</p>
          </div>
          <span className="px-3 py-1 rounded-full bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-400 font-mono text-xs font-bold w-fit">
            76.56 → 6.64 PPL Drop!
          </span>
        </div>

        <div className="h-64 sm:h-72 w-full pt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis
                dataKey="shortName"
                tick={{ fill: "#64748b", fontSize: 11 }}
                axisLine={{ stroke: "#cbd5e1" }}
              />
              <YAxis
                tick={{ fill: "#64748b", fontSize: 11 }}
                axisLine={{ stroke: "#cbd5e1" }}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const item = payload[0].payload;
                    return (
                      <div className="bg-slate-900 text-white p-3 rounded-xl border border-slate-700 shadow-xl text-xs space-y-1">
                        <p className="font-bold">{item.name}</p>
                        <p className="text-emerald-400 font-mono">Perplexity (PPL): {item.ppl}</p>
                        <p className="text-slate-400 font-mono">Val Loss: {item.loss}</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="ppl" radius={[8, 8, 0, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}
