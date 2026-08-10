"use client";

import { useState } from "react";
import { Play, Sparkles, RefreshCw, Cpu, Layers, CheckCircle2, Sliders } from "lucide-react";

export default function DemoPage() {
  const [prompt, setPrompt] = useState("KING HENRY:\nShall I be bold to tell you what I think?");
  const [selectedModel, setSelectedModel] = useState<"nanollm-211k" | "gpt2-124m">("nanollm-211k");
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(20);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(100);
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState("");
  const [activeSource, setActiveSource] = useState("");

  const handleGenerate = async () => {
    setLoading(true);
    setOutput("");

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          model: selectedModel,
          temperature,
          top_k: topK,
          top_p: topP,
          max_tokens: maxTokens,
        }),
      });

      const data = await res.json();
      if (data.text) {
        setOutput(data.text);
        setActiveSource(data.source || selectedModel);
      } else {
        setOutput("Error: " + (data.error || "Failed to generate text."));
      }
    } catch (err: any) {
      setOutput("Error connecting to API endpoint.");
    } finally {
      setLoading(false);
    }
  };

  const presets = [
    { label: "King Henry", text: "KING HENRY:\nShall I be bold to tell you what I think?" },
    { label: "Romeo & Juliet", text: "ROMEO:\nLady, by yonder blessed moon I vow" },
    { label: "AI Philosophy", text: "The secret to artificial intelligence is" },
  ];

  return (
    <div className="space-y-8 py-6">
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-purple-100 dark:bg-purple-950/80 border border-purple-300 dark:border-purple-800/60 text-purple-700 dark:text-purple-400 text-xs font-semibold">
          <Play className="w-3.5 h-3.5" />
          <span>Live Dual-Model Inference Playground</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 dark:text-white">Live Model Demo</h1>
        <p className="text-slate-600 dark:text-slate-400 text-sm max-w-2xl">
          Test real-time autoregressive text completion. Switch between your custom 211k Mini-GPT (built from scratch) and fine-tuned 124M GPT-2!
        </p>
      </div>

      {/* Model Selector Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
        <button
          onClick={() => setSelectedModel("nanollm-211k")}
          className={`p-5 rounded-2xl border text-left transition-all ${
            selectedModel === "nanollm-211k"
              ? "bg-indigo-50 dark:bg-indigo-950/60 border-indigo-500 shadow-md ring-2 ring-indigo-500/20"
              : "bg-white dark:bg-slate-900/60 border-slate-300 dark:border-slate-800 hover:border-slate-400"
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="px-2.5 py-1 rounded-md bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300 font-mono text-[10px] font-bold">
              FROM SCRATCH
            </span>
            {selectedModel === "nanollm-211k" && <CheckCircle2 className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />}
          </div>
          <h3 className="font-bold text-slate-900 dark:text-white text-base pt-2">NanoLLM Mini-GPT</h3>
          <p className="text-xs text-slate-600 dark:text-slate-400 pt-1">211,777 parameters | $|V| = 65$ / BPE | Fast local inference</p>
        </button>

        <button
          onClick={() => setSelectedModel("gpt2-124m")}
          className={`p-5 rounded-2xl border text-left transition-all ${
            selectedModel === "gpt2-124m"
              ? "bg-purple-50 dark:bg-purple-950/60 border-purple-500 shadow-md ring-2 ring-purple-500/20"
              : "bg-white dark:bg-slate-900/60 border-slate-300 dark:border-slate-800 hover:border-slate-400"
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="px-2.5 py-1 rounded-md bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 font-mono text-[10px] font-bold">
              FOUNDATION MODEL
            </span>
            {selectedModel === "gpt2-124m" && <CheckCircle2 className="w-5 h-5 text-purple-600 dark:text-purple-400" />}
          </div>
          <h3 className="font-bold text-slate-900 dark:text-white text-base pt-2">Fine-Tuned GPT-2</h3>
          <p className="text-xs text-slate-600 dark:text-slate-400 pt-1">124,439,808 parameters | $|V_{BPE}| = 50,257$ | Hugging Face Cloud</p>
        </button>
      </div>

      {/* Main Playground Interface */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Controls Column */}
        <div className="bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 rounded-2xl p-6 space-y-6 shadow-sm">
          
          {/* Prompt Presets */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">Prompt Presets</label>
            <div className="flex flex-wrap gap-2">
              {presets.map((preset, idx) => (
                <button
                  key={idx}
                  onClick={() => setPrompt(preset.text)}
                  className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 text-xs font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-800 transition-all"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          {/* Text Input */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">Input Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
              placeholder="Type your prompt here..."
              className="w-full bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-xl p-3 text-xs text-slate-900 dark:text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
            />
          </div>

          {/* Sliders */}
          <div className="space-y-4 pt-2 border-t border-slate-200 dark:border-slate-800">
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-slate-700 dark:text-slate-300">Temperature (T): {temperature}</span>
                <span className="text-slate-500 text-[11px]">{temperature <= 0.5 ? "Conservative" : temperature <= 1.0 ? "Balanced" : "Creative"}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="2.0"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-indigo-600"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">Top-K Filter: {topK}</span>
                <input
                  type="range"
                  min="1"
                  max="100"
                  step="1"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full accent-purple-600"
                />
              </div>

              <div className="space-y-1">
                <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">Top-P Nucleus: {topP}</span>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value))}
                  className="w-full accent-pink-600"
                />
              </div>
            </div>
          </div>

          <button
            onClick={handleGenerate}
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-3.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-xs font-bold hover:opacity-95 transition-all shadow-md shadow-indigo-500/20"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            <span>{loading ? "Generating Text..." : "Generate Completion"}</span>
          </button>
        </div>

        {/* Output Column */}
        <div className="bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 rounded-2xl p-6 space-y-3 shadow-sm flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">Generated Text Completion</label>
              {activeSource && (
                <span className="px-2.5 py-0.5 rounded-full bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 font-mono text-[10px] font-bold">
                  {activeSource}
                </span>
              )}
            </div>

            <div className="w-full h-80 bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-xl p-4 font-mono text-xs text-indigo-900 dark:text-indigo-300 overflow-y-auto whitespace-pre-wrap leading-relaxed">
              {loading ? (
                <div className="h-full flex flex-col items-center justify-center space-y-3 text-slate-400">
                  <RefreshCw className="w-6 h-6 animate-spin text-indigo-500" />
                  <span className="text-xs font-sans">Sampling autoregressive logits...</span>
                </div>
              ) : output ? (
                output
              ) : (
                <span className="text-slate-400 italic">Click "Generate Completion" to run real-time inference...</span>
              )}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
