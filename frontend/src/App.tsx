import { useEffect, useRef, useState } from "react";
import { ChatWindow } from "./components/ChatWindow";
import { MessageInput } from "./components/MessageInput";
import { Sidebar } from "./components/Sidebar";
import { createThread, deleteThread, getThreadMessages, listThreads, renameThread, streamChat } from "./lib/api";
import type { Message, ProgressStep, Thread } from "./lib/types";
import "./App.css";

const USER_ID_STORAGE_KEY = "trade-agent.user-id";

// Abort a chat request that has produced no event for this long.
const STALL_MS = 120_000;

export default function App() {
  const [userId, setUserId] = useState(
    () => localStorage.getItem(USER_ID_STORAGE_KEY) ?? "swapnik",
  );
  const [threads, setThreads] = useState<Thread[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(false);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingAssistant, setStreamingAssistant] = useState<string | null>(null);
  const [steps, setSteps] = useState<ProgressStep[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  // Mobile only: whether the conversations sidebar is slid open as a drawer.
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // The text of the last message that failed to get a reply — kept so the
  // error bubble can offer a one-click retry instead of leaving the user
  // stuck re-typing. Note: retrying resends the message rather than
  // resuming the dropped run, so the user's text may end up persisted
  // twice in the thread if the first attempt's user message already made
  // it to the backend before failing — acceptable trade-off for now.
  const [lastFailedMessage, setLastFailedMessage] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  // Persist the chosen user id and reload their threads whenever it changes.
  useEffect(() => {
    localStorage.setItem(USER_ID_STORAGE_KEY, userId);
    if (!userId) return;
    setThreadsLoading(true);
    setActiveThreadId(null);
    setMessages([]);
    listThreads(userId)
      .then(setThreads)
      .catch((e) => setError(String(e)))
      .finally(() => setThreadsLoading(false));
  }, [userId]);

  const selectThread = async (threadId: string) => {
    setActiveThreadId(threadId);
    setError(null);
    setStreamingAssistant(null);
    setSteps([]);
    try {
      const history = await getThreadMessages(threadId);
      setMessages(history);
    } catch (e) {
      setError(String(e));
    }
  };

  const newThread = async () => {
    setError(null);
    try {
      const thread = await createThread(userId);
      setThreads((prev) => [thread, ...prev]);
      setActiveThreadId(thread.id);
      setMessages([]);
    } catch (e) {
      setError(String(e));
    }
  };

  const renameThreadTitle = async (threadId: string, title: string) => {
    try {
      const updated = await renameThread(threadId, title);
      setThreads((prev) => prev.map((t) => (t.id === threadId ? updated : t)));
    } catch (e) {
      setError(String(e));
    }
  };

  const deleteThreadById = async (threadId: string) => {
    try {
      await deleteThread(threadId);
      setThreads((prev) => prev.filter((t) => t.id !== threadId));
      if (activeThreadId === threadId) {
        setActiveThreadId(null);
        setMessages([]);
      }
    } catch (e) {
      setError(String(e));
    }
  };

  const sendMessage = async (text: string) => {
    if (!activeThreadId) return;

    setError(null);
    setLastFailedMessage(null);
    setSending(true);
    setSteps([]);
    setStreamingAssistant("");

    // Optimistic local echo of the user's message; the canonical copy
    // (with a real id/timestamp) is fetched from the backend once the
    // turn completes.
    const optimisticUserMessage: Message = {
      id: `local-${Date.now()}`,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimisticUserMessage]);

    const controller = new AbortController();
    abortRef.current = controller;

    // A turn is only complete if the server said so ("done" or "error").
    // If the stream just ends, or goes silent for STALL_MS (a dropped
    // connection, a stalled model call), report it instead of showing nothing.
    let finished = false;
    let stalled = false;
    let watchdog: ReturnType<typeof setTimeout> | undefined;
    const armWatchdog = () => {
      if (watchdog) clearTimeout(watchdog);
      watchdog = setTimeout(() => {
        stalled = true;
        controller.abort();
      }, STALL_MS);
    };
    armWatchdog();

    try {
      for await (const event of streamChat({
        message: text,
        userId,
        threadId: activeThreadId,
        signal: controller.signal,
      })) {
        armWatchdog();
        if (event.type === "status") {
          setSteps((prev) => {
            if (event.phase === "start") {
              const key = event.id ?? `${event.tool}-${prev.length}`;
              return prev.some((s) => s.key === key)
                ? prev
                : [...prev, { key, tool: event.tool, nested: event.nested, done: false }];
            }
            // finish: mark the matching step (by id, else last unfinished same-tool step)
            let marked = false;
            return [...prev]
              .reverse()
              .map((s) => {
                if (!marked && !s.done && (event.id ? s.key === event.id : s.tool === event.tool)) {
                  marked = true;
                  return { ...s, done: true };
                }
                return s;
              })
              .reverse();
          });
        } else if (event.type === "assistant_delta") {
          setSteps([]);
          // Append this chunk onto whatever's streamed in so far, rather
          // than replacing — assistant_delta events are incremental.
          setStreamingAssistant((prev) => (prev ?? "") + event.content);
        } else if (event.type === "error") {
          finished = true;
          setError(event.content);
          setLastFailedMessage(text);
        } else if (event.type === "done") {
          finished = true;
        }
      }

      if (!finished) {
        setError("The connection ended before the answer finished. Please retry.");
        setLastFailedMessage(text);
      }

      // Re-sync from the server so ids/timestamps are canonical and we
      // pick up exactly what got persisted — including the auto-generated
      // title the backend sets on a thread's first message.
      const [history, refreshedThreads] = await Promise.all([
        getThreadMessages(activeThreadId),
        listThreads(userId),
      ]);
      setMessages(history);
      setThreads(refreshedThreads);
    } catch (e) {
      setError(
        stalled
          ? "No response for a while, so the request was stopped. Please retry."
          : String(e),
      );
      setLastFailedMessage(text);
      // Show what the server did persist (the question, maybe a partial reply).
      void getThreadMessages(activeThreadId).then(setMessages).catch(() => {});
    } finally {
      if (watchdog) clearTimeout(watchdog);
      setStreamingAssistant(null);
      setSteps([]);
      setSending(false);
      abortRef.current = null;
    }
  };

  const retryLastMessage = () => {
    if (!lastFailedMessage) return;
    const text = lastFailedMessage;
    setLastFailedMessage(null);
    void sendMessage(text);
  };

  const activeTitle = threads.find((t) => t.id === activeThreadId)?.title;

  return (
    <div className="app-layout">
      <Sidebar
        open={sidebarOpen}
        userId={userId}
        onChangeUser={setUserId}
        threads={threads}
        activeThreadId={activeThreadId}
        onSelectThread={(id) => {
          setSidebarOpen(false);
          void selectThread(id);
        }}
        onNewThread={() => {
          setSidebarOpen(false);
          void newThread();
        }}
        onRenameThread={renameThreadTitle}
        onDeleteThread={deleteThreadById}
        loading={threadsLoading}
      />

      {/* Mobile only (hidden on desktop via CSS): tap-away backdrop for the drawer. */}
      {sidebarOpen && <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />}

      <main className="main-panel">
        {/* Mobile only (hidden on desktop via CSS): header with the drawer toggle. */}
        <header className="mobile-header">
          <button
            className="menu-btn"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open conversations"
          >
            ☰
          </button>
          <div className="mobile-title">{activeTitle || "Trade Agent"}</div>
        </header>
        {activeThreadId ? (
          <>
            <ChatWindow
              messages={messages}
              streamingAssistant={streamingAssistant}
              steps={steps}
              sending={sending}
              error={error}
              onRetry={lastFailedMessage ? retryLastMessage : undefined}
            />
            <MessageInput onSend={sendMessage} disabled={sending} />
          </>
        ) : (
          <div className="no-thread-selected">
            Select a conversation, or start a new one, to begin.
          </div>
        )}
      </main>
    </div>
  );
}
