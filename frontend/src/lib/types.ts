export interface Thread {
  id: string;
  user_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export type ChatEvent =
  | { type: "tool_result"; tool: string; content: string }
  // One token (or small batch of tokens) of the assistant's reply. The
  // consumer appends these together to build the full text — this is NOT
  // the full message each time, unlike the old "assistant" event it
  // replaced.
  | { type: "assistant_delta"; content: string }
  // Live progress: a tool call started or finished. `nested` means it ran
  // inside a sub-agent (portfolio analyst / answer verifier).
  | { type: "status"; phase: "start" | "done"; id: string | null; tool: string; nested: boolean }
  | { type: "done" }
  | { type: "error"; content: string };

export interface ProgressStep {
  key: string;
  tool: string;
  nested: boolean;
  done: boolean;
}
