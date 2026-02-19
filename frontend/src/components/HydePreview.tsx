// src/components/HydePreview.tsx

"use client";

import { Sparkles, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

interface HydePreviewProps {
  abstract: string;
}

export default function HydePreview({ abstract }: HydePreviewProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-xl border border-brand-500/20 bg-brand-500/5">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-4 py-2.5 text-left"
      >
        <Sparkles className="h-4 w-4 text-brand-400 shrink-0" />
        <span className="text-sm font-medium text-brand-300">
          HyDE Hypothetical Abstract
        </span>
        <span className="text-xs text-[var(--text-muted)] ml-1">
          — AI-generated ideal paper used for better matching
        </span>
        <span className="ml-auto">
          {open ? (
            <ChevronUp className="h-3 w-3 text-brand-400" />
          ) : (
            <ChevronDown className="h-3 w-3 text-brand-400" />
          )}
        </span>
      </button>

      {open && (
        <div className="border-t border-brand-500/10 px-4 py-3 animate-slideDown">
          <p className="text-sm leading-relaxed text-[var(--text-secondary)] italic">
            &ldquo;{abstract}&rdquo;
          </p>
        </div>
      )}
    </div>
  );
}
