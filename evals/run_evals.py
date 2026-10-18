#!/usr/bin/env python3
"""Run the eval cases against a deployed (or local) backend.

Each case: create a fresh thread, POST the prompt to /chat, read the SSE
stream, join the answer text, then check it against cases.py. Standard
library only, so nothing to install.

  export EVAL_BASE_URL=https://trade-agent-backend-boog.onrender.com
  export EVAL_API_KEY=...            # the backend's API_KEY (X-API-Key header)
  python evals/run_evals.py --list   # show the cases, call nothing
  python evals/run_evals.py          # run everything (uses model quota!)
  python evals/run_evals.py --only buy_nvda,trade_by_id
  python evals/run_evals.py --skip-heavy   # leave out the portfolio review

Optional, with LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (and LANGFUSE_HOST):
  --langfuse   look up each run's trace to report tokens, latency and Groq
               errors, check expected tools were called, and attach an
               eval_pass score to the trace.

Exit code: 0 all passed, 1 some failed, 2 nothing could run (e.g. quota or
backend down). A quota/rate-limit error is reported as QUOTA, not FAIL.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cases import CASES, Case, check_text  # noqa: E402

QUOTA_RE = re.compile(r"quota|resource_exhausted|rate.?limit|too large|\b429\b|\b413\b", re.I)


@dataclass
class Result:
    case: Case
    status: str  # PASS | FAIL | QUOTA | ERROR
    seconds: float = 0.0
    chars: int = 0
    reasons: list[str] = field(default_factory=list)
    answer: str = ""
    thread_id: str = ""
    trace: dict = field(default_factory=dict)


# ---------------------------------------------------------------- HTTP

def _request(url, *, method="GET", body=None, headers=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    return urllib.request.Request(url, data=data, method=method, headers=hdrs), timeout


def http_json(url, *, method="GET", body=None, headers=None, timeout=60):
    req, timeout = _request(url, method=method, body=body, headers=headers, timeout=timeout)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else {}


def wake_backend(base, wait_s=120):
    """Render's free instances sleep; poll /health until it answers."""
    deadline = time.time() + wait_s
    while True:
        try:
            http_json(f"{base}/health", timeout=20)
            return True
        except Exception as exc:  # noqa: BLE001
            if time.time() > deadline:
                print(f"backend did not answer /health: {exc}", file=sys.stderr)
                return False
            time.sleep(5)


def stream_chat(base, headers, thread_id, user_id, prompt, timeout=240):
    """Yield parsed SSE events ({"type": ..., "content": ...}) from /chat."""
    req, _ = _request(
        f"{base}/chat",
        method="POST",
        body={"message": prompt, "user_id": user_id, "thread_id": thread_id},
        headers={**headers, "Accept": "text/event-stream"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue  # event:, comments (: ping) and blank lines
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------- Langfuse

class Langfuse:
    def __init__(self):
        self.host = os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com").rstrip("/")
        pk, sk = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
        if not (pk and sk):
            raise RuntimeError("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
        token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        self.headers = {"Authorization": f"Basic {token}"}

    def trace_for_thread(self, thread_id, wait_s=60):
        """The backend sets langfuse_session_id = thread_id. Traces arrive
        asynchronously, so poll for a bit."""
        deadline = time.time() + wait_s
        while time.time() < deadline:
            listing = http_json(
                f"{self.host}/api/public/traces?sessionId={thread_id}&limit=5",
                headers=self.headers,
            )
            if listing.get("data"):
                trace_id = listing["data"][0]["id"]
                return http_json(f"{self.host}/api/public/traces/{trace_id}", headers=self.headers)
            time.sleep(5)
        return None

    @staticmethod
    def summarize(trace):
        obs = trace.get("observations") or []

        def tokens(o):
            u = o.get("usageDetails") or o.get("usage") or {}
            return int(u.get("total") or u.get("totalTokens") or 0)

        gens = [o for o in obs if o.get("type") == "GENERATION"]
        return {
            "trace_id": trace.get("id"),
            "latency_s": trace.get("latency"),
            "tokens": sum(tokens(o) for o in gens),
            "tools": sorted({o.get("name") for o in obs if o.get("type") == "TOOL" and o.get("name")}),
            "groq_errors": sum(
                1 for o in gens if "groq" in (o.get("name") or "").lower() and o.get("level") == "ERROR"
            ),
            "gemini_calls": sum(1 for o in gens if "googlegenerativeai" in (o.get("name") or "").lower()),
        }

    def score(self, trace_id, name, value, comment=""):
        http_json(
            f"{self.host}/api/public/scores",
            method="POST",
            body={"traceId": trace_id, "name": name, "value": value, "comment": comment[:500]},
            headers=self.headers,
        )


# ---------------------------------------------------------------- running

def run_case(case, base, headers, user_id) -> Result:
    start = time.time()
    res = Result(case=case, status="ERROR")
    try:
        thread = http_json(
            f"{base}/threads", method="POST",
            body={"user_id": user_id, "title": f"eval: {case.id}"}, headers=headers,
        )
        res.thread_id = thread["id"]
        parts: list[str] = []
        error = None
        finished = False
        for ev in stream_chat(base, headers, res.thread_id, user_id, case.prompt):
            kind = ev.get("type")
            if kind == "assistant_delta":
                parts.append(ev.get("content", ""))
            elif kind == "error":
                error = ev.get("content", "unknown error")
            elif kind == "done":
                finished = True
        res.answer = "".join(parts)
        res.chars = len(res.answer)
        if error:
            res.status = "QUOTA" if QUOTA_RE.search(error) else "ERROR"
            res.reasons = [f"stream error: {error[:200]}"]
        elif not finished:
            res.status, res.reasons = "ERROR", ["stream ended without a done event"]
        else:
            res.reasons = check_text(case, res.answer)
            res.status = "FAIL" if res.reasons else "PASS"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        res.status = "QUOTA" if exc.code in (413, 429) else "ERROR"
        res.reasons = [f"HTTP {exc.code}: {detail}"]
    except Exception as exc:  # noqa: BLE001
        res.reasons = [f"{type(exc).__name__}: {exc}"]
    res.seconds = time.time() - start
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("EVAL_API_KEY", ""))
    ap.add_argument("--user-id", default=os.environ.get("EVAL_USER_ID", "eval-runner"))
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--skip-heavy", action="store_true", help="skip cases that use the sub-agent")
    ap.add_argument("--delay", type=float, default=8.0, help="seconds between cases (default 8)")
    ap.add_argument("--heavy-wait", type=float, default=60.0,
                    help="extra pause before heavy cases so rate-limit windows reset (default 60)")
    ap.add_argument("--langfuse", action="store_true", help="read each run's trace and attach scores")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--list", action="store_true", help="print the cases and exit (no calls)")
    args = ap.parse_args()

    cases = list(CASES)
    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        unknown = wanted - {c.id for c in cases}
        if unknown:
            print(f"unknown case ids: {sorted(unknown)}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c.id in wanted]
    if args.skip_heavy:
        cases = [c for c in cases if not c.heavy]
    cases.sort(key=lambda c: c.heavy)  # heavy last

    if args.list:
        for c in cases:
            print(f"{c.id:20} {'(heavy) ' if c.heavy else '        '}{c.prompt}")
        return 0

    if not args.base_url:
        print("set EVAL_BASE_URL (or --base-url)", file=sys.stderr)
        return 2
    base = args.base_url.rstrip("/")
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    lf = None
    if args.langfuse:
        try:
            lf = Langfuse()
        except Exception as exc:  # noqa: BLE001
            print(f"[langfuse] disabled: {exc}", file=sys.stderr)

    print(f"waking {base} ...")
    if not wake_backend(base):
        return 2

    results: list[Result] = []
    for i, case in enumerate(cases):
        if i:
            time.sleep(args.heavy_wait if case.heavy else args.delay)
        print(f"[{i + 1}/{len(cases)}] {case.id} ...", flush=True)
        res = run_case(case, base, headers, args.user_id)
        if lf and res.thread_id and res.status in ("PASS", "FAIL"):
            try:
                trace = lf.trace_for_thread(res.thread_id)
                if trace:
                    res.trace = lf.summarize(trace)
                    missing = [t for t in case.expect_tools if t not in res.trace["tools"]]
                    if missing:
                        res.reasons.append(f"tools not called: {missing}")
                        res.status = "FAIL"
                    lf.score(res.trace["trace_id"], "eval_pass", 1 if res.status == "PASS" else 0,
                             "; ".join(res.reasons))
                else:
                    res.reasons.append("no Langfuse trace found (tool checks skipped)")
            except Exception as exc:  # noqa: BLE001
                res.reasons.append(f"langfuse lookup failed: {exc}")
        results.append(res)
        print(f"    {res.status}  {res.seconds:.1f}s  {res.chars} chars"
              + (f"  | {'; '.join(res.reasons)}" if res.reasons else ""))
        if res.status == "QUOTA":
            print("    (quota/rate limit: not counted as a regression)")

    print("\nSummary")
    print(f"{'case':20} {'status':6} {'secs':>6} {'chars':>6} {'tokens':>7} {'groq_err':>8} {'gemini':>6}")
    for r in results:
        t = r.trace
        print(f"{r.case.id:20} {r.status:6} {r.seconds:6.1f} {r.chars:6d} "
              f"{t.get('tokens', ''):>7} {t.get('groq_errors', ''):>8} {t.get('gemini_calls', ''):>6}")
    n = {s: sum(1 for r in results if r.status == s) for s in ("PASS", "FAIL", "QUOTA", "ERROR")}
    print(f"\n{n['PASS']} passed, {n['FAIL']} failed, {n['QUOTA']} quota, {n['ERROR']} errors "
          f"(of {len(results)})")

    with open(args.out, "w") as fh:
        json.dump(
            [{"id": r.case.id, "status": r.status, "seconds": round(r.seconds, 1), "chars": r.chars,
              "reasons": r.reasons, "trace": r.trace, "thread_id": r.thread_id, "answer": r.answer}
             for r in results],
            fh, indent=2,
        )
    print(f"details written to {args.out}")

    if n["FAIL"]:
        return 1
    if n["PASS"] == 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
