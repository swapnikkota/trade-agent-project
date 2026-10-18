"""
deepagents agent factory, instrumented with Langfuse.
"""
import copy
import functools
import logging
from typing import Any, List, Optional

from deepagents import HarnessProfile, create_deep_agent, register_harness_profile
from langfuse.langchain import CallbackHandler

from app.config import Settings

logger = logging.getLogger("trade_agent")

TRADE_AGENT_INSTRUCTIONS = """You are a trading data assistant.

You have access to tools backed by an MCP server that can look up trade
information. When the user asks about trades, positions, prices, or related
data, use the available tools to fetch accurate, current information rather
than guessing.

Available tools and when to use them:
- get_trades(symbol?, side?, limit?, offset?) — raw trade list, optionally
  filtered. Use this for "show me trades for X" / "BUY trades for Y".
- get_trade_by_id(trade_id) — a single trade by its exact ID.
- create_trade(symbol, side, quantity, price) — records a new trade. Only
  call this when the user explicitly asks to log/record/create a trade,
  never speculatively.
- get_trade_summary() — portfolio-wide overview: total trades, BUY/SELL
  breakdown, top symbols, value by symbol. Use for "summarize my trades" /
  "how's my portfolio looking".
- get_largest_trades(top_n?) — the biggest trades by total value. Use for
  "what are my biggest trades".
- get_trades_by_symbol_analysis(symbol) — one symbol's full picture: avg
  buy/sell price, net position, long/short status. Use for "how am I doing
  on AAPL" / "what's my TSLA position".
- detect_large_trades(threshold_value?) — flags trades over a dollar
  threshold. Use for "any big trades" / "flag anything over $X".
- get_recent_trades(n?) — the N most recent trades by time. Use for
  "what happened recently" / "last 10 trades" when no symbol/side filter
  is implied.

Prefer the more specific analysis tool over hand-rolling the same
computation from get_trades yourself — e.g. for "my AAPL position" call
get_trades_by_symbol_analysis("AAPL") rather than calling get_trades and
computing net position inline; the dedicated tool is tested and consistent.
You can still call more than one tool in a turn when a question genuinely
needs it (e.g. "summarize my trades and flag anything big" → both
get_trade_summary and detect_large_trades).

Formatting: show trade timestamps as "YYYY-MM-DD HH:MM UTC" (drop seconds,
fractions and the trailing "Z"), e.g. 2026-10-02 17:38 UTC.

Be concise and precise with numbers. If a query is ambiguous (e.g. missing a
ticker, date range, or account), ask a clarifying question before calling a
tool.

Accuracy rules — follow these strictly:
- Every trade you mention must be attributed to the exact "symbol" field
  returned by the tool for that specific trade. Never assume, infer, or
  carry over a symbol from an earlier trade or an earlier turn in the
  conversation.
- Before including a trade in your answer, re-check that its "symbol"
  field actually matches what the user asked about. If the user asked
  about one ticker (e.g. MSFT), only report trades whose "symbol" field
  is exactly that ticker — never a different one you happened to also
  retrieve.
- If a tool call returns trades for multiple symbols, treat each trade
  independently by its own "symbol" field. Do not blend details (price,
  quantity, date) from one trade into another, even if they were
  returned in the same tool call.
- If you are not fully certain a trade matches the symbol the user asked
  about, say so explicitly rather than stating it as fact.

Calling get_trades correctly — this is critical:
- The "side" parameter accepts exactly ONE value: "BUY" or "SELL" — or it
  must be omitted entirely. It does NOT accept a combined value.
- If the user does not specify a side (e.g. "trades for TSLA", "any
  trades for Apple"), you MUST omit the "side" parameter completely from
  the tool call so the tool returns trades of both sides. Do NOT pass
  something like "BUY, SELL", "BUY/SELL", "BOTH", or any other combined
  or made-up value for "side" — that will match neither real value and
  silently return zero results, even when matching trades exist.
- The same applies to "symbol": pass it only when the user named a
  specific ticker. Don't invent a combined or placeholder value for any
  parameter just because you want to be more specific — an omitted
  parameter is the correct way to say "no filter on this field."

Handling empty or missing tool results — this is critical:
- If a tool call returns an empty result (e.g. an empty list, no matching
  records), you MUST tell the user no matching trades were found. Say
  something like "No sell trades found for MSFT" — do not soften this
  into an invented answer.
- Never invent, guess, or state specific trade details (price, quantity,
  date, symbol) that did not come directly from a tool result you
  received in this exact turn. A trade you are not looking at right now
  does not exist for the purposes of your answer.
- An empty result is a normal, valid, useful answer. It is never a
  reason to make something up instead.
"""

# Appended to the main prompt only when the matching sub-agent is enabled
# (see build_agent), so the agent is never told to call one that isn't there.
PORTFOLIO_ANALYST_DELEGATION = """

Delegating a full portfolio review — use the `task` tool with
subagent_type="portfolio_analyst" when the user asks for a broad review
("full portfolio review", "analyze my portfolio", "give me a report on my
trades", "how am I doing overall"). It runs the summary, largest-trade,
large-trade and per-symbol analysis tools itself (for every symbol in the
portfolio) and hands back one complete report. Present that report to the
user as your answer, keeping its labels and numbers as given (a symbol's
"total value" is gross traded value — buys plus sells — never buy-minus-sell),
and do NOT call any other tool afterwards — it already
covers every symbol, so re-running lookups only adds delay. Do NOT use it
for simple lookups ("show my AAPL trades", "last 5 trades") — answer those
directly with the tools above.
"""

VERIFIER_DELEGATION = """

Verifying your answer — before you give the user a final answer that states
specific trades (any symbol, side, quantity, price, or date for an individual
trade), call the `task` tool with subagent_type="answer_verifier". In the
task description include (1) the user's question and (2) your complete draft
answer. It re-checks every figure against fresh tool data and replies either
"VERIFIED" or a list of discrepancies. If it lists discrepancies, correct
your answer to match the verified data before replying, and do not mention
the verification step to the user unless a correction changes the answer.
Skip the verifier for answers that state no individual trade details (e.g.
"no matching trades found", clarifying questions) and for create_trade
confirmations.
Also skip it when your answer is just the portfolio_analyst's report passed
along — the analyst copies its figures straight from the same tools the
verifier would re-query, so checking it again only adds delay. The verifier
is for answers you assembled yourself from direct lookups (get_trades,
get_recent_trades, get_trade_by_id, ...), where a symbol or price mix-up can
slip in.
"""

PORTFOLIO_ANALYST_PROMPT = """You are a portfolio analyst working from a trade database.

You will be given a request for a portfolio review. Work in as few turns as
possible by making independent tool calls TOGETHER in a single turn (parallel
tool calls), not one at a time:

Turn 1 — call all three at once:
- get_trade_summary() for the overall picture.
- get_largest_trades(top_n=5) for the biggest positions by value.
- detect_large_trades() to flag unusually large trades. Call it with NO
  arguments so the tool's own default threshold is used — never pass a
  threshold of your own.

Turn 2 — using the symbols in the summary's value_by_symbol (up to 6, highest
total value first), call get_trades_by_symbol_analysis(symbol) for EVERY one
of them, all at once. Do not skip a symbol: a symbol with only SELL trades is
a SHORT position and must appear in the report.

Then reply with ONE concise report: overall stats, the largest trades,
anything flagged as unusually large, and a net-position line for EVERY symbol
(include LONG/SHORT status). A symbol's "total value" is GROSS traded value
(buys + sells added together) — label it "gross traded value" and never
describe it as buy-minus-sell or net. Copy numbers exactly as the tools returned them — never
round differently, estimate, or fill gaps. If a tool returns nothing, say
so in the report rather than guessing. Show any timestamp as "YYYY-MM-DD HH:MM UTC" (no seconds or fractions).
Report facts only: no investment
advice or predictions. Your reply goes to another agent, not the end user,
so skip pleasantries and keep it compact.
"""

ANSWER_VERIFIER_PROMPT = """You are a strict fact-checker for a trading-data assistant.

You will be given a user's question and a DRAFT ANSWER that states specific
trades. Check it against fresh data:
1. Work out which trades/figures the draft claims (symbol, side, quantity,
   price, date, totals, net positions).
2. Re-fetch the underlying data yourself with the read-only tools
   (get_trades, get_trade_by_id, get_recent_trades, get_trades_by_symbol_analysis,
   etc.) — do not trust the draft.
3. Compare every claim to the tool output. Each trade must carry the exact
   "symbol", "side", "quantity" and "price" the tool returned for THAT trade;
   flag any trade that blends details from different trades, belongs to a
   different symbol than stated, or does not exist in the data.

Reply in exactly one of two forms:
- "VERIFIED" — every claim matches the data.
- "DISCREPANCIES:" followed by one line per problem, each stating what the
  draft said and what the data actually shows.

Do not rewrite the answer, add commentary, or suggest anything else.
"""

# Read-only allowlist for sub-agents. An allowlist (not "everything except
# create_trade") so a write tool added to the MCP server later is NOT
# silently available to a sub-agent; add new read tools here deliberately.
READ_ONLY_TOOL_NAMES = frozenset(
    {
        "get_trades",
        "get_trade_by_id",
        "get_trade_summary",
        "get_largest_trades",
        "get_trades_by_symbol_analysis",
        "detect_large_trades",
        "get_recent_trades",
    }
)


def normalize_empty_tool_results(tools: List[Any]) -> List[Any]:
    """
    Makes an empty MCP tool result safe to send back to the model.

    A tool that finds nothing (e.g. get_trades with no matching rows) comes
    back from langchain-mcp-adapters as an EMPTY LIST of content blocks.
    Groq rejects that outright when it's replayed as the tool message on the
    next model call: HTTP 400 "for 'role:tool' ... minimum number of items
    is 1". The main agent never hit this because /chat short-circuits on an
    empty result before another model call happens — but sub-agents (the
    answer_verifier re-fetching data that doesn't exist) have no such guard,
    and neither does deepagents' built-in general-purpose sub-agent.

    Fixing it on the tools themselves covers every agent that uses them.
    "[]" is what _is_empty_tool_result in main.py already treats as "no
    results", so the main agent's behaviour is unchanged.
    """
    for t in tools:
        original = getattr(t, "coroutine", None)
        if original is None or getattr(original, "_empty_result_normalized", False):
            continue

        @functools.wraps(original)
        async def wrapped(*args: Any, __original: Any = original, **kwargs: Any) -> Any:
            # Models often send an explicit null for optional params they
            # mean to omit (see make_optional_args_nullable); the MCP server
            # would reject null for a number, so drop them: omitted == default.
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            result = await __original(*args, **kwargs)
            # content_and_artifact tools return (content, artifact).
            if isinstance(result, tuple) and len(result) == 2:
                content, artifact = result
                if isinstance(content, list) and not content:
                    return "[]", artifact
            return result

        wrapped._empty_result_normalized = True  # type: ignore[attr-defined]
        t.coroutine = wrapped
    return tools


def make_optional_args_nullable(tools: List[Any]) -> List[Any]:
    """
    Lets optional tool parameters be null in the schema sent to the model.

    Groq validates the model's tool call against the schema server-side and
    rejects it outright ("parameters for tool detect_large_trades did not
    match schema: /threshold_value: expected number, but got null") when the
    model passes null for an optional number it meant to omit. That aborted
    a whole chat turn. Accepting null here makes such calls valid; the null
    is then dropped before the MCP call (see normalize_empty_tool_results).
    Required parameters are left strict.
    """
    for t in tools:
        schema = getattr(t, "args_schema", None)
        if not isinstance(schema, dict):
            continue
        schema = copy.deepcopy(schema)
        required = set(schema.get("required", []))
        for name, prop in (schema.get("properties") or {}).items():
            if name in required or not isinstance(prop, dict):
                continue
            if prop.get("type") == "null" or any(
                isinstance(o, dict) and o.get("type") == "null" for o in prop.get("anyOf", [])
            ):
                continue
            outer = {k: prop[k] for k in ("description", "default", "title") if k in prop}
            inner = {k: v for k, v in prop.items() if k not in outer}
            schema["properties"][name] = {**outer, "anyOf": [inner, {"type": "null"}]}
        t.args_schema = schema
    return tools


def build_langfuse_handler() -> CallbackHandler:
    """
    Langfuse v3's CallbackHandler reads credentials from the process
    environment (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST),
    not constructor args — those env vars are set in main.py via
    load_dotenv() at startup. user_id/session_id are attached per-call via
    the "langfuse_user_id" / "langfuse_session_id" metadata keys instead of
    being baked into the handler itself.
    """
    return CallbackHandler()


def build_model(settings: Settings, model_name: Optional[str] = None):
    """
    Builds the chat model deepagents will run on, based on
    settings.model_provider. Swapping providers is a config change only —
    the rest of the agent/tool wiring stays the same.
    """
    if settings.model_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    elif settings.model_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )
    elif settings.model_provider == "groq":
        from app.agent.groq_retry import RetryingChatGroq as ChatGroq

        groq_model = model_name or settings.groq_model

        if settings.llm_gateway_url:
            # Same ChatGroq client (same tool-calling/reasoning handling),
            # just pointed at the gateway. The Groq SDK appends
            # /openai/v1/... to the base URL; the gateway ignores the path
            # and forwards to Groq (see gateway/config.example.yaml).
            return ChatGroq(
                model=groq_model,
                api_key=settings.llm_gateway_api_key,
                groq_api_base=settings.llm_gateway_url.rstrip("/"),
                temperature=0,
            )
        return ChatGroq(
            model=groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )
    else:
        raise ValueError(f"Unsupported model_provider: {settings.model_provider}")


def build_gemini(settings: Settings):
    """Gemini chat model (direct to Google). Imported lazily so the package is
    only needed when a Gemini option is switched on."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0,
    )


def _gemini_enabled(settings: Settings, wanted: bool, what: str) -> bool:
    if not wanted:
        return False
    if not settings.gemini_api_key:
        logger.warning("%s=gemini but GEMINI_API_KEY is empty: ignoring", what)
        return False
    return True


def _fallback_middleware(settings: Settings) -> List[Any]:
    """ModelFallbackMiddleware that re-runs a failed model call on Gemini.

    Fresh instance per agent (main and each sub-agent); returns [] when off.
    """
    if not _gemini_enabled(settings, settings.fallback_provider == "gemini", "FALLBACK_PROVIDER"):
        return []
    from langchain.agents.middleware import ModelFallbackMiddleware

    return [ModelFallbackMiddleware(build_gemini(settings))]


# deepagents ships filesystem/scratchpad tools this app never uses. Their
# schemas are re-sent on every model call (~2k tokens), which makes Groq's
# per-minute token limit bite sooner. A HarnessProfile hides them from the
# model (main agent and sub-agents); `task` and our MCP tools are untouched.
UNUSED_BUILTIN_TOOLS = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "write_todos", "execute"}
)


def hide_unused_builtin_tools(settings: Settings) -> None:
    if not settings.hide_builtin_file_tools:
        return
    keep = {n.strip() for n in settings.keep_builtin_tools.split(",") if n.strip()}
    register_harness_profile(
        settings.model_provider,
        HarnessProfile(excluded_tools=UNUSED_BUILTIN_TOOLS - keep),
    )


def build_agent(tools: List[Any], settings: Settings):
    """
    Creates the deepagents agent wired to the given tools (from MCP) and
    the configured model provider. Sub-agents, etc. get extended here in
    later phases.
    """
    hide_unused_builtin_tools(settings)
    model = build_model(settings)
    tools = make_optional_args_nullable(normalize_empty_tool_results(tools))

    system_prompt = TRADE_AGENT_INSTRUCTIONS
    subagents: List[Any] = []

    if settings.subagents_enabled:
        read_only_tools = [t for t in tools if t.name in READ_ONLY_TOOL_NAMES]

        # Sub-agents inherit the main agent's model unless SUBAGENT_MODEL is
        # set (Groq only), which gives them their own rate-limit bucket.
        # Tools are explicit and read-only: neither can call create_trade.
        subagent_extra: dict = {}
        sub_on_gemini = False
        if _gemini_enabled(settings, settings.subagent_provider == "gemini", "SUBAGENT_PROVIDER"):
            subagent_extra["model"] = build_gemini(settings)
            sub_on_gemini = True
        elif settings.subagent_model and settings.model_provider == "groq":
            subagent_extra["model"] = build_model(settings, settings.subagent_model)

        def _sub_fallback() -> dict:
            # A fresh middleware per sub-agent; pointless if it already runs on Gemini.
            mw = [] if sub_on_gemini else _fallback_middleware(settings)
            return {"middleware": mw} if mw else {}

        subagents.append(
            {
                "name": "portfolio_analyst",
                "description": (
                    "Produces a full portfolio review report (overall stats, largest "
                    "trades, unusually large trades, per-symbol net positions) by running "
                    "several analysis tools. Use for broad 'review/analyze my portfolio' "
                    "requests, not for simple lookups."
                ),
                "system_prompt": PORTFOLIO_ANALYST_PROMPT,
                "tools": read_only_tools,
                **subagent_extra,
                **_sub_fallback(),
            }
        )
        system_prompt += PORTFOLIO_ANALYST_DELEGATION

        if settings.verifier_enabled:
            subagents.append(
                {
                    "name": "answer_verifier",
                    "description": (
                        "Fact-checks a draft answer that states specific trades by "
                        "re-fetching the data. Pass the user's question and the full "
                        "draft answer. Replies 'VERIFIED' or a list of discrepancies."
                    ),
                    "system_prompt": ANSWER_VERIFIER_PROMPT,
                    "tools": read_only_tools,
                    **subagent_extra,
                    **_sub_fallback(),
                }
            )
            system_prompt += VERIFIER_DELEGATION

    agent = create_deep_agent(
        tools=tools,
        system_prompt=system_prompt,
        model=model,
        subagents=subagents or None,
        middleware=_fallback_middleware(settings),
    )
    return agent
