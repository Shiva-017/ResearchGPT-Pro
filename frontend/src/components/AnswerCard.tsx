// src/components/AnswerCard.tsx

"use client";

import { MessageSquare } from "lucide-react";
import { useMemo } from "react";

interface AnswerCardProps {
  answer: string;
  onCitationClick?: (index: number) => void;
}

/**
 * Render the LLM answer with basic markdown:
 *  **bold**, [N] citation badges, bullet points, newlines
 */
function renderAnswer(text: string) {
  const lines = text.split("\n");

  return lines.map((line, li) => {
    // Bullet points
    const isBullet = /^\s*[-•*]\s+/.test(line);
    const cleanLine = isBullet ? line.replace(/^\s*[-•*]\s+/, "") : line;

    // Process inline formatting
    const parts = cleanLine.split(/(\*\*[^*]+\*\*|\[\d+\])/g);
    const rendered = parts.map((part, pi) => {
      // Bold
      if (part.startsWith("**") && part.endsWith("**")) {
        return (
          <strong key={pi} className="text-[var(--text-primary)] font-semibold">
            {part.slice(2, -2)}
          </strong>
        );
      }
      // Citation [N]
      const citMatch = part.match(/^\[(\d+)\]$/);
      if (citMatch) {
        return (
          <span
            key={pi}
            className="inline-flex items-center justify-center mx-0.5 h-5 min-w-[20px] rounded bg-brand-600/20 px-1.5 text-[11px] font-bold text-brand-400 cursor-default"
            title={`Paper ${citMatch[1]}`}
          >
            {citMatch[1]}
          </span>
        );
      }
      return part;
    });

    if (isBullet) {
      return (
        <li key={li} className="ml-4 list-disc text-sm leading-relaxed text-[var(--text-secondary)]">
          {rendered}
        </li>
      );
    }

    if (cleanLine.trim() === "") {
      return <br key={li} />;
    }

    return (
      <p key={li} className="text-sm leading-relaxed text-[var(--text-secondary)]">
        {rendered}
      </p>
    );
  });
}

export default function AnswerCard({ answer }: AnswerCardProps) {
  const content = useMemo(() => renderAnswer(answer), [answer]);

  return (
    <div className="rounded-xl border border-brand-500/20 bg-gradient-to-br from-[var(--bg-card)] to-brand-500/[0.03]">
      {/* Header */}
      <div className="flex items-center gap-2.5 px-5 pt-4 pb-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-600/20">
          <MessageSquare className="h-4 w-4 text-brand-400" />
        </div>
        <span className="text-sm font-semibold text-brand-300">
          ResearchGPT Answer
        </span>
      </div>

      {/* Answer body */}
      <div className="px-5 pb-5 space-y-1.5">
        {content}
      </div>
    </div>
  );
}
