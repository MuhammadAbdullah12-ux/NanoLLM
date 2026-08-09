import { User, CheckCircle2 } from "lucide-react";

function GithubIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
      <path d="M9 18c-4.51 2-5-2-7-2" />
    </svg>
  );
}

export default function AboutPage() {
  const skills = [
    "PyTorch & Deep Learning Architecture Design",
    "Transformer Self-Attention & Positional Encoding",
    "Subword Byte-Pair Encoding Tokenization (tiktoken)",
    "Cosine LR Annealing, AdamW & Regularization",
    "Hugging Face Transformers & Model Fine-Tuning",
    "Full-Stack Web Development (Next.js, React, Tailwind)"
  ];

  return (
    <div className="space-y-8 py-6 max-w-4xl mx-auto">
      <div className="space-y-3 text-center sm:text-left">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-950/80 border border-indigo-800/60 text-indigo-400 text-xs font-semibold">
          <User className="w-3.5 h-3.5" />
          <span>Freelance AI & Software Engineer</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white">About the Author</h1>
        <p className="text-slate-400 text-sm max-w-2xl">
          Hi! I'm Muhammad Abdullah. I specialize in building custom AI architectures, fine-tuning foundation models, and deploying modern full-stack web applications.
        </p>
      </div>

      <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 sm:p-8 space-y-6">
        <h2 className="text-xl font-bold text-white">Core Technical Expertise</h2>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {skills.map((skill, idx) => (
            <div key={idx} className="flex items-center gap-2 text-xs text-slate-300 p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
              <span>{skill}</span>
            </div>
          ))}
        </div>

        <div className="pt-4 border-t border-slate-800 flex flex-wrap gap-4">
          <a
            href="https://github.com/MuhammadAbdullah12-ux/NanoLLM"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition-all shadow-md shadow-indigo-500/20"
          >
            <GithubIcon className="w-4 h-4" />
            <span>Explore GitHub Profile</span>
          </a>
        </div>
      </div>
    </div>
  );
}
