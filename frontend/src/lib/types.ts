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
  | { type: "done" }
  | { type: "error"; content: string };
