"""
deepagents agent factory, instrumented with Langfuse.
"""
from typing import Any, List

from deepagents import create_deep_agent
from langfuse.langchain import CallbackHandler

from app.config import Settings

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


def build_model(settings: Settings):
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
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )
    else:
        raise ValueError(f"Unsupported model_provider: {settings.model_provider}")


def build_agent(tools: List[Any], settings: Settings):
    """
    Creates the deepagents agent wired to the given tools (from MCP) and
    the configured model provider. Sub-agents, etc. get extended here in
    later phases.
    """
    model = build_model(settings)
    agent = create_deep_agent(
        tools=tools,
        system_prompt=TRADE_AGENT_INSTRUCTIONS,
        model=model,
    )
    return agent
