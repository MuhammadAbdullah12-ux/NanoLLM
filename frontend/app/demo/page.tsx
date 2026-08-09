"use client";

import { useState } from "react";
import { Play, Sparkles, Sliders, RefreshCw } from "lucide-react";

export default function DemoPage() {
  const [prompt, setPrompt] = useState("KING HENRY:\nShall I be bold to tell you what I think?");
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(100);
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState("");

  const handleGenerate = async () => {
    setLoading(true);
    // Simulated demo generation output for interactive playground
    setTimeout(() => {
      setOutput(
        `${prompt}\n\nKING HENRY:\nTo tell you what I think: that is the noble question!\nBy yonder blessed moon, we shall march forward unto the breach,\nWhere courage speaks and honor guides our hearts.`
      );
      setLoading(false);
    }, 800);
  };

  return (
    <div className="space-y-8 py-6">
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-950/80 border border-purple-800/60 text-purple-400 text-xs font-semibold">
          <Play className="w-3.5 h-3.5" />
          <span>Interactive Model Playground</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white">Live Generation Demo</h1>
        <p className="text-slate-400 text-sm max-w-2xl">
          Test real-time autoregressive text completion with temperature scaling and nucleus sampling.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 pt-2">
        
        {/* Input Controls */}
        <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 space-y-6">
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-300">Input Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
            />
          </div>

          <div className="space-y-4">
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Temperature (T): {temperature}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="2.0"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-indigo-500"
            />
          </div>

          <button
            onClick={handleGenerate}
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-xs font-bold hover:opacity-95 transition-all shadow-md shadow-indigo-500/20"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            <span>{loading ? "Generating Text..." : "Generate Text Completion"}</span>
          </button>
        </div>

        {/* Output Box */}
        <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 space-y-3">
          <label className="text-xs font-semibold text-slate-300">Generated Text Completion</label>
          <div className="w-full h-64 bg-slate-950 border border-slate-800 rounded-xl p-4 font-mono text-xs text-indigo-300 overflow-y-auto whitespace-pre-wrap">
            {output || <span className="text-slate-600 italic">Click "Generate Text Completion" to view output...</span>}
          </div>
        </div>

      </div>
    </div>
  );
}
