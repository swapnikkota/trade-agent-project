import type { ChatEvent, Message, Thread } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8080";
// Phase 6: shared-secret API key, sent on every request via X-API-Key.
// Undefined/empty here is fine as long as the backend's API_KEY is also
// unset (auth off, the local-dev default) — if the backend has a key
// configured and this doesn't match it, every request gets a 401.
const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  return {
    ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
    ...extra,
  };
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function createThread(userId: string, title?: string): Promise<Thread> {
  const res = await fetch(`${API_BASE}/threads`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ user_id: userId, title: title ?? null }),
  });
  return jsonOrThrow<Thread>(res);
}

export async function listThreads(userId: string): Promise<Thread[]> {
  const res = await fetch(`${API_BASE}/threads?user_id=${encodeURIComponent(userId)}`, {
    headers: authHeaders(),
  });
  return jsonOrThrow<Thread[]>(res);
}

export async function getThreadMessages(threadId: string): Promise<Message[]> {
  const res = await fetch(`${API_BASE}/threads/${threadId}/messages`, {
    headers: authHeaders(),
  });
  return jsonOrThrow<Message[]>(res);
}

export async function renameThread(threadId: string, title: string): Promise<Thread> {
  const res = await fetch(`${API_BASE}/threads/${threadId}`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ title }),
  });
  return jsonOrThrow<Thread>(res);
}

export async function deleteThread(threadId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/threads/${threadId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
}

/**
 * Streams a /chat response as an async generator of parsed ChatEvents.
 *
 * The backend uses sse-starlette, which emits standard SSE frames
 * ("event: message\ndata: {...}\n\n"). We can't use the native
 * EventSource here because it only supports GET requests, and /chat is a
 * POST — so we read the raw fetch body stream and parse SSE frames by
 * hand instead.
 */
export async function* streamChat(params: {
  message: string;
  userId: string;
  threadId: string;
  signal?: AbortSignal;
}): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      message: params.message,
      user_id: params.userId,
      thread_id: params.threadId,
    }),
    signal: params.signal,
  });

  if (!res.ok || !res.body) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }

  function parseFrame(frame: string): ChatEvent | null {
    const dataLines = frame
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim());
    if (dataLines.length === 0) return null;

    const raw = dataLines.join("\n");
    // sse-starlette sends periodic ": ping - <timestamp>" comment
    // lines to keep the connection alive; these aren't "data:" lines
    // so they're already filtered out above, but guard anyway.
    if (!raw || raw.startsWith(":")) return null;

    try {
      return JSON.parse(raw) as ChatEvent;
    } catch {
      // Malformed frame — ignore rather than killing the whole stream.
      return null;
    }
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? ""; // keep the last (possibly incomplete) frame

    for (const frame of frames) {
      const parsed = parseFrame(frame);
      if (parsed) yield parsed;
    }
  }

  // The loop above only emits a frame once a trailing blank-line
  // terminator ("\n\n") has been seen, and holds back whatever comes
  // after it in `buffer` in case more data is still coming. If the
  // server closes the connection right after its very last event (e.g.
  // an error aborts the run and there's nothing more to stream), that
  // final frame can end up sitting in `buffer` with no further read()
  // call ever arriving to trigger its processing — silently dropping
  // exactly the event that mattered most. Flush it here once the stream
  // has genuinely ended.
  const finalParsed = parseFrame(buffer);
  if (finalParsed) yield finalParsed;
}
