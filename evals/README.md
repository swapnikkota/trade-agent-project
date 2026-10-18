# Evals

A small, repeatable check that the agent still answers correctly after a
change (model, prompt, gateway, tool settings). It talks to the backend over
HTTP exactly like the frontend: one fresh thread per case, `POST /chat`, read
the streamed answer, check it.

## Run

```bash
export EVAL_BASE_URL=https://trade-agent-backend-boog.onrender.com
export EVAL_API_KEY=...        # the backend's API_KEY (sent as X-API-Key); don't commit it

python evals/run_evals.py --list           # show cases, makes no calls
python evals/run_evals.py --skip-heavy     # the 9 light cases (Groq only)
python evals/run_evals.py                  # all 10, incl. the portfolio review
python evals/run_evals.py --only buy_nvda,trade_by_id
```

Run it from your own machine (the backend URL must be reachable). It uses only
the Python standard library.

## Quota

Every case is a real model call. The portfolio review also uses the Gemini
sub-agent, which on the free tier allows about 20 requests a day. So:

- Run `--skip-heavy` for routine checks; run the full set sparingly.
- Cases are spaced out (`--delay`, default 8s; `--heavy-wait`, default 60s
  before the portfolio review) to stay under rate limits.
- A quota or rate-limit error is reported as `QUOTA`, not `FAIL`.

## What is checked

- The stream finishes with `done` (no `error` event).
- The answer text contains the right facts (numbers, symbols, order) and
  none of the wrong ones, and stays under a length cap. See `cases.py`.
- Expected values come from the `TRADES` list in `cases.py`. **If the data in
  trades-api changes, update that list.**

## Optional: Langfuse

```bash
export LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...   # LANGFUSE_HOST defaults to us.cloud.langfuse.com
python evals/run_evals.py --langfuse
```

Looks up each run's trace (the backend uses the thread id as the Langfuse
session id) to report tokens, Groq errors and Gemini calls, check that
`expect_tools` were called, and attach an `eval_pass` score to the trace.
This part is written against Langfuse's public API from its documented shape
and has not been run against your project yet; if a lookup fails the run
carries on and notes it.

Results are written to `eval_results.json` (answers included). Exit code:
0 all passed, 1 some failed, 2 nothing could run.
