import type { Message } from "../lib/types";

interface ChatWindowProps {
  messages: Message[];
  streamingAssistant: string | null;
  toolCall: string | null;
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
    create_trade: "Recording trade…",
    // market_data MCP server (Phase 5 test server)
    get_quote: "Checking the latest quote…",
  };
  return known[tool] ?? `Using ${tool}…`;
}

export function ChatWindow({ messages, streamingAssistant, toolCall, error, onRetry }: ChatWindowProps) {
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
            <div className={`bubble ${m.role}`}>{m.content}</div>
          </div>
        ))}

        {toolCall && (
          <div className="bubble-row assistant">
            <div className="bubble tool-indicator">{friendlyToolLabel(toolCall)}</div>
          </div>
        )}

        {showStreamingBubble && (
          <div className="bubble-row assistant">
            <div className="bubble assistant">{streamingAssistant}</div>
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
