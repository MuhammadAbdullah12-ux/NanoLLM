import { User, CheckCircle2, Mail, ExternalLink, Award, Sparkles, Cpu, Layers } from "lucide-react";

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
    "Full-Stack Web Development (Next.js 14, React, Tailwind CSS)",
    "Serverless REST API Integration & Vercel Deployment",
    "Quantitative Model Benchmarking & Perplexity Evaluation"
  ];

  return (
    <div className="space-y-10 py-6 max-w-4xl mx-auto">
      
      {/* Header Bio */}
      <div className="space-y-4 text-center sm:text-left">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-indigo-100 dark:bg-indigo-950/80 border border-indigo-300 dark:border-indigo-800/60 text-indigo-700 dark:text-indigo-400 text-xs font-semibold">
          <User className="w-3.5 h-3.5" />
          <span>Freelance AI & Full-Stack Software Engineer</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-extrabold text-slate-900 dark:text-white">Muhammad Abdullah</h1>
        <p className="text-slate-600 dark:text-slate-300 text-base max-w-2xl leading-relaxed">
          I specialize in building custom Deep Learning architectures from scratch in PyTorch, fine-tuning large foundation models, and deploying modern full-stack web applications on Vercel and cloud platforms.
        </p>
      </div>

      {/* Freelance Offerings Card */}
      <div className="bg-gradient-to-r from-indigo-900 to-purple-900 text-white rounded-3xl p-8 space-y-4 shadow-xl">
        <div className="flex items-center gap-2 text-indigo-300 text-xs font-mono font-bold uppercase tracking-wider">
          <Sparkles className="w-4 h-4 text-indigo-400" />
          <span>Freelance Services</span>
        </div>
        <h2 className="text-2xl font-bold">Looking for Custom AI or Full-Stack Engineering?</h2>
        <p className="text-sm text-indigo-100 leading-relaxed max-w-2xl">
          Whether you need a custom Small Language Model trained on domain data, fine-tuning of open-source foundation models (LLaMA, GPT-2, Mistral), or a high-end Next.js web platform, I deliver production-ready code with complete transparency.
        </p>
        <div className="pt-2 flex flex-wrap gap-4">
          <a
            href="https://github.com/MuhammadAbdullah12-ux"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-5 py-3 rounded-xl bg-white text-indigo-950 font-bold text-xs hover:bg-slate-100 transition-all shadow-md"
          >
            <GithubIcon className="w-4 h-4 text-indigo-900" />
            <span>GitHub Profile</span>
          </a>
        </div>
      </div>

      {/* Core Technical Skills */}
      <div className="bg-white dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800/80 rounded-2xl p-6 sm:p-8 space-y-6 shadow-sm">
        <h2 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Award className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
          <span>Core Technical Competencies</span>
        </h2>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {skills.map((skill, idx) => (
            <div key={idx} className="flex items-center gap-2 text-xs font-medium text-slate-800 dark:text-slate-200 p-3.5 rounded-xl bg-slate-100 dark:bg-slate-950/80 border border-slate-300 dark:border-slate-800">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
              <span>{skill}</span>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
