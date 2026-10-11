import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message, ProgressStep } from "../lib/types";

interface ChatWindowProps {
  messages: Message[];
  streamingAssistant: string | null;
  steps: ProgressStep[];
  sending: boolean;
  error: string | null;
  // Present only when there's a failed message to resend; its absence
  // hides the Retry button rather than rendering it disabled.
  onRetry?: () => void;
}

function friendlyToolLabel(tool: string): string {
  // Small, hand-maintained map for the tools we know about; anything
  // unrecognized falls back to a generic phrasing rather than exposing
  // the raw function name.
  const known: Record<string, string> = {
    // trade MCP server
    get_trades: "Looking up trades…",
    get_trade_by_id: "Looking up trade details…",
    get_trade_summary: "Summarizing your portfolio…",
    get_largest_trades: "Finding your largest trades…",
    get_trades_by_symbol_analysis: "Analyzing trades by symbol…",
    detect_large_trades: "Scanning for unusually large trades…",
    get_recent_trades: "Checking recent trades…",
    create_trade: "Recording trade…",
    // deepagents sub-agents (portfolio analyst / answer verifier)
    task: "Delegating to a specialist…",
    // market_data MCP server (Phase 5 test server)
    get_quote: "Checking the latest quote…",
  };
  return known[tool] ?? `Using ${tool}…`;
}

function ProgressPanel({ steps }: { steps: ProgressStep[] }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const started = Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(id);
  }, []);

  const current = [...steps].reverse().find((s) => !s.done);
  const trail = steps.filter((s) => s.done).slice(-4);
  // After ~20s with nothing new it's most likely a rate-limit wait on the model.
  const label = current ? friendlyToolLabel(current.tool) : steps.length ? "Thinking…" : "Starting…";

  return (
    <div className="bubble-row assistant">
      <div className="bubble tool-indicator progress-panel">
        <div className="progress-current">
          <span className="progress-spinner" aria-hidden="true" />
          <span>{label}</span>
          <span className="progress-elapsed">{elapsed}s</span>
        </div>
        {trail.length > 0 && (
          <ul className="progress-trail">
            {trail.map((s) => (
              <li key={s.key}>✓ {friendlyToolLabel(s.tool).replace(/…$/, "")}</li>
            ))}
          </ul>
        )}
        {elapsed >= 20 && <div className="progress-hint">Still working — the model may be rate-limited.</div>}
      </div>
    </div>
  );
}

function Markdown({ text }: { text: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}

export function ChatWindow({ messages, streamingAssistant, steps, sending, error, onRetry }: ChatWindowProps) {
  const showStreamingBubble = streamingAssistant !== null && streamingAssistant.length > 0;
  const isEmpty = messages.length === 0 && !showStreamingBubble;

  if (isEmpty) {
    return (
      <div className="chat-window chat-window-empty">
        <div className="empty-state">Ask something about your trades to get started.</div>
      </div>
    );
  }

  return (
    <div className="chat-window">
      <div className="chat-window-inner">
        {messages.map((m) => (
          <div key={m.id} className={`bubble-row ${m.role}`}>
            <div className={`bubble ${m.role}`}>
              {m.role === "assistant" ? <Markdown text={m.content} /> : m.content}
            </div>
          </div>
        ))}

        {sending && !showStreamingBubble && <ProgressPanel steps={steps} />}

        {showStreamingBubble && (
          <div className="bubble-row assistant">
            <div className="bubble assistant">
              <Markdown text={streamingAssistant ?? ""} />
            </div>
          </div>
        )}

        {error && (
          <div className="bubble-row assistant">
            <div className="bubble error">
              <div>⚠ {error}</div>
              {onRetry && (
                <button className="retry-btn" onClick={onRetry}>
                  Retry
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
