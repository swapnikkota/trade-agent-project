"""
Translate raw LangGraph stream chunks into small "status" events for the UI.

The UI wants to show what the agent is doing *while* a query runs (which
tool it is calling, including the tools sub-agents call inside the `task`
tool). With `subgraphs=True`, LangGraph also yields sub-agent chunks, tagged
with a non-empty namespace tuple; top-level chunks have an empty one.

Only tool *starts* (an AI chunk announcing a tool call by name) and tool
*finishes* (a ToolMessage) are reported. Text tokens are never reported here.
"""
from typing import Optional


def status_event(namespace: tuple, message_chunk) -> Optional[dict]:
    """Return a status payload for this chunk, or None if it isn't a tool boundary."""
    nested = len(namespace) > 0
    msg_type = getattr(message_chunk, "type", None)

    if msg_type == "tool":
        return {
            "type": "status",
            "phase": "done",
            "id": getattr(message_chunk, "tool_call_id", None),
            "tool": getattr(message_chunk, "name", None) or "tool",
            "nested": nested,
        }

    if msg_type in ("ai", "AIMessageChunk"):
        # The first streamed chunk of a tool call carries its name and id;
        # later chunks only carry argument fragments (name is empty).
        for tc in getattr(message_chunk, "tool_call_chunks", None) or []:
            if tc.get("name"):
                return {
                    "type": "status",
                    "phase": "start",
                    "id": tc.get("id"),
                    "tool": tc["name"],
                    "nested": nested,
                }
    return None
