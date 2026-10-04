import { useState, type KeyboardEvent, type MouseEvent } from "react";
import type { Thread } from "../lib/types";

interface SidebarProps {
  userId: string;
  onChangeUser: (userId: string) => void;
  threads: Thread[];
  activeThreadId: string | null;
  onSelectThread: (threadId: string) => void;
  onNewThread: () => void;
  onRenameThread: (threadId: string, title: string) => void;
  onDeleteThread: (threadId: string) => void;
  loading: boolean;
}

export function Sidebar({
  userId,
  onChangeUser,
  threads,
  activeThreadId,
  onSelectThread,
  onNewThread,
  onRenameThread,
  onDeleteThread,
  loading,
}: SidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");

  const startEditing = (t: Thread) => {
    setEditingId(t.id);
    setDraftTitle(t.title ?? "");
  };

  const commitEdit = () => {
    if (editingId && draftTitle.trim()) {
      onRenameThread(editingId, draftTitle.trim());
    }
    setEditingId(null);
  };

  const handleEditKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commitEdit();
    } else if (e.key === "Escape") {
      setEditingId(null);
    }
  };

  const handleDelete = (e: MouseEvent, t: Thread) => {
    e.stopPropagation();
    const label = t.title || "this conversation";
    if (window.confirm(`Delete "${label}"? This can't be undone.`)) {
      onDeleteThread(t.id);
    }
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <label className="user-label">
          User
          <input
            className="user-input"
            value={userId}
            onChange={(e) => onChangeUser(e.target.value)}
            placeholder="user id"
          />
        </label>
        <button className="new-thread-btn" onClick={onNewThread} disabled={!userId}>
          + New chat
        </button>
      </div>

      <div className="thread-list">
        {loading && <div className="thread-list-status">Loading…</div>}
        {!loading && threads.length === 0 && (
          <div className="thread-list-status">No conversations yet</div>
        )}
        {threads.map((t) => (
          <div
            key={t.id}
            className={`thread-item ${t.id === activeThreadId ? "active" : ""}`}
            role="button"
            tabIndex={0}
            onClick={() => editingId !== t.id && onSelectThread(t.id)}
            onKeyDown={(e) => {
              if (editingId !== t.id && (e.key === "Enter" || e.key === " ")) {
                e.preventDefault();
                onSelectThread(t.id);
              }
            }}
          >
            <div className="thread-item-row">
              {editingId === t.id ? (
                <input
                  autoFocus
                  className="thread-title-edit"
                  value={draftTitle}
                  onChange={(e) => setDraftTitle(e.target.value)}
                  onBlur={commitEdit}
                  onKeyDown={handleEditKeyDown}
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <div
                  className="thread-title"
                  onDoubleClick={(e) => {
                    e.stopPropagation();
                    startEditing(t);
                  }}
                  title="Double-click to rename"
                >
                  {t.title || "Untitled conversation"}
                </div>
              )}
              <button
                className="thread-delete-btn"
                onClick={(e) => handleDelete(e, t)}
                title="Delete conversation"
                aria-label="Delete conversation"
              >
                ×
              </button>
            </div>
            <div className="thread-meta">{new Date(t.updated_at).toLocaleString()}</div>
          </div>
        ))}
      </div>
    </aside>
  );
}
