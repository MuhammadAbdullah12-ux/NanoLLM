"use client";

import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";

export default function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    // Check initial saved theme or system preference
    const savedTheme = localStorage.getItem("nanollm_theme") as "dark" | "light" | null;
    if (savedTheme) {
      setTheme(savedTheme);
      document.documentElement.classList.toggle("dark", savedTheme === "dark");
    } else {
      document.documentElement.classList.add("dark");
    }
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    localStorage.setItem("nanollm_theme", nextTheme);

    if (nextTheme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  return (
    <button
      onClick={toggleTheme}
      aria-label="Toggle Day and Night Theme"
      className="relative flex items-center justify-between w-14 h-7 p-1 rounded-full bg-slate-800 dark:bg-slate-900 border border-slate-700/80 dark:border-slate-800 shadow-inner transition-colors duration-300 focus:outline-none"
    >
      {/* Moving Thumb Indicator */}
      <span
        className={`w-5 h-5 rounded-full flex items-center justify-center transition-transform duration-300 shadow-md ${
          theme === "dark"
            ? "translate-x-7 bg-indigo-600 text-indigo-100"
            : "translate-x-0 bg-amber-400 text-amber-950"
        }`}
      >
        {theme === "dark" ? (
          <Moon className="w-3 h-3 fill-indigo-200" />
        ) : (
          <Sun className="w-3 h-3 fill-amber-950" />
        )}
      </span>

      {/* Background Icons */}
      <span className="absolute left-1.5 text-amber-400 opacity-60 dark:opacity-40">
        <Sun className="w-3.5 h-3.5" />
      </span>
      <span className="absolute right-1.5 text-indigo-400 opacity-40 dark:opacity-60">
        <Moon className="w-3.5 h-3.5" />
      </span>
    </button>
  );
}
