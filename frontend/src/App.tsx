import { useEffect, useRef, useState } from "react";
import { ChatWindow } from "./components/ChatWindow";
import { MessageInput } from "./components/MessageInput";
import { Sidebar } from "./components/Sidebar";
import { createThread, deleteThread, getThreadMessages, listThreads, renameThread, streamChat } from "./lib/api";
import type { Message, Thread } from "./lib/types";
import "./App.css";

const USER_ID_STORAGE_KEY = "trade-agent.user-id";

export default function App() {
  const [userId, setUserId] = useState(
    () => localStorage.getItem(USER_ID_STORAGE_KEY) ?? "swapnik",
  );
  const [threads, setThreads] = useState<Thread[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(false);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingAssistant, setStreamingAssistant] = useState<string | null>(null);
  const [toolCall, setToolCall] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
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
    setToolCall(null);
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
    setToolCall(null);
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

    try {
      for await (const event of streamChat({
        message: text,
        userId,
        threadId: activeThreadId,
        signal: controller.signal,
      })) {
        if (event.type === "tool_result") {
          setToolCall(event.tool);
        } else if (event.type === "assistant_delta") {
          setToolCall(null);
          // Append this chunk onto whatever's streamed in so far, rather
          // than replacing — assistant_delta events are incremental.
          setStreamingAssistant((prev) => (prev ?? "") + event.content);
        } else if (event.type === "error") {
          setError(event.content);
          setLastFailedMessage(text);
        } else if (event.type === "done") {
          // no-op; handled after the loop
        }
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
      setError(String(e));
      setLastFailedMessage(text);
    } finally {
      setStreamingAssistant(null);
      setToolCall(null);
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

  return (
    <div className="app-layout">
      <Sidebar
        userId={userId}
        onChangeUser={setUserId}
        threads={threads}
        activeThreadId={activeThreadId}
        onSelectThread={selectThread}
        onNewThread={newThread}
        onRenameThread={renameThreadTitle}
        onDeleteThread={deleteThreadById}
        loading={threadsLoading}
      />

      <main className="main-panel">
        {activeThreadId ? (
          <>
            <ChatWindow
              messages={messages}
              streamingAssistant={streamingAssistant}
              toolCall={toolCall}
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
