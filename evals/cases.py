"""Eval cases for the trading agent.

The expected values come from TRADES below: the five trades the deployed
database held when these cases were written (the numbers were cross-checked
against a full portfolio review). If the data in trades-api changes, update
TRADES: the cases that depend on specific numbers are derived from it.

Matching is intentionally forgiving about formatting ($7,000 / 7000 / 7,000.00)
and strict about the facts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TRADES = [
    # id, symbol, side, quantity, price
    (1, "AAPL", "BUY", 10.0, 195.50),
    (2, "TSLA", "SELL", 5.0, 245.00),
    (3, "NVDA", "BUY", 8.0, 875.00),
    (4, "MSFT", "BUY", 15.0, 415.00),
    (5, "AAPL", "SELL", 3.0, 198.00),
]


def value(trade) -> float:
    return trade[3] * trade[4]


def money(x: float) -> str:
    """Regex matching an amount in common formats: 7000, 7,000, 7000.00,
    7,000.00, and 195.5 / 195.50."""
    whole = int(x)
    with_comma = f"{whole:,}".replace(",", ",?")
    cents = int(round((x - whole) * 100))
    if cents == 0:
        return rf"{with_comma}(\.0+)?"
    digits = f"{cents:02d}"
    if digits[1] == "0":
        return rf"{with_comma}\.{digits[0]}0?"
    return rf"{with_comma}\.{digits}"


def net_shares(symbol: str) -> float:
    return sum(t[3] if t[2] == "BUY" else -t[3] for t in TRADES if t[1] == symbol)


def gross_value(symbol: str) -> float:
    return sum(value(t) for t in TRADES if t[1] == symbol)


@dataclass
class Case:
    id: str
    prompt: str
    # Every regex here must match the answer (case-insensitive).
    must_match: list[str] = field(default_factory=list)
    # None of these may match.
    must_not_match: list[str] = field(default_factory=list)
    # These must appear in this order.
    in_order: list[str] = field(default_factory=list)
    max_chars: int = 3000
    # Tool names the trace should show (only checked with --langfuse).
    expect_tools: list[str] = field(default_factory=list)
    # Heavy cases use the sub-agent (and Gemini): run last, after a pause.
    heavy: bool = False


def _sym(s: str) -> str:
    return rf"\b{s}\b"


CASES: list[Case] = [
    Case(
        id="recent_trades",
        prompt="Show me the last 5 trades",
        must_match=[_sym(s) for s in ("AAPL", "TSLA", "NVDA", "MSFT")]
        + [money(t[4]) for t in TRADES],
    ),
    Case(
        id="buy_nvda",
        prompt="Show me the BUY trades for NVDA",
        must_match=[_sym("NVDA"), money(875.0), r"\b8(\.0+)?\b"],
        must_not_match=[_sym("TSLA")],
    ),
    Case(
        id="sell_trades",
        prompt="Show me all my SELL trades",
        must_match=[_sym("TSLA"), _sym("AAPL"), money(245.0), money(198.0)],
        must_not_match=[_sym("NVDA"), _sym("MSFT")],
    ),
    Case(
        id="trade_by_id",
        prompt="Show me trade 3",
        must_match=[_sym("NVDA"), money(875.0), r"\bBUY\b"],
        must_not_match=[_sym("TSLA")],
        expect_tools=["get_trade_by_id"],
    ),
    Case(
        id="net_position_aapl",
        prompt="What is my net position in AAPL?",
        must_match=[_sym("AAPL"), rf"\b{int(net_shares('AAPL'))}(\.0+)?\b"],
    ),
    Case(
        id="net_position_tsla",
        prompt="What is my net position in TSLA?",
        must_match=[_sym("TSLA"), r"(-|−|‑)\s?5(\.0+)?\b|short|sold"],
    ),
    Case(
        id="largest_three",
        prompt="What are my 3 largest trades by total value?",
        in_order=[money(value(TRADES[2])), money(value(TRADES[3])), money(value(TRADES[0]))],
        must_match=[_sym("NVDA"), _sym("MSFT"), _sym("AAPL")],
        expect_tools=["get_largest_trades"],
    ),
    Case(
        id="unknown_symbol",
        prompt="Show me my trades for GOOG",
        must_match=[r"\bno\b|\bnone\b|not found|couldn.t find|don.t (see|have)|did not find"],
        must_not_match=[r"GOOG[^.\n]{0,40}\$\s?\d"],
    ),
    Case(
        id="large_trades",
        prompt="Are there any unusually large trades in my history?",
        must_match=[r"\bno\b|\bnone\b|not any|no trades", r"10,?000|10\s?k"],
    ),
    Case(
        id="portfolio_review",
        prompt="Give me a full portfolio review",
        must_match=[
            money(gross_value("NVDA")),
            money(gross_value("MSFT")),
            money(gross_value("AAPL")),
            money(gross_value("TSLA")),
            r"\b5\b",
            _sym("NVDA"),
            _sym("MSFT"),
            _sym("AAPL"),
            _sym("TSLA"),
        ],
        max_chars=6000,
        expect_tools=["get_trade_summary", "get_largest_trades"],
        heavy=True,
    ),
]


def check_text(case: Case, text: str) -> list[str]:
    """Return a list of failure reasons (empty = pass)."""
    problems: list[str] = []
    if not text.strip():
        return ["empty answer"]
    if len(text) > case.max_chars:
        problems.append(f"answer is {len(text)} chars (limit {case.max_chars})")
    for pat in case.must_match:
        if not re.search(pat, text, re.IGNORECASE):
            problems.append(f"missing /{pat}/")
    for pat in case.must_not_match:
        if re.search(pat, text, re.IGNORECASE):
            problems.append(f"unexpected /{pat}/")
    if case.in_order:
        pos = 0
        for pat in case.in_order:
            m = re.search(pat, text[pos:], re.IGNORECASE)
            if not m:
                problems.append(f"/{pat}/ not found in the expected order")
                break
            pos += m.end()
    return problems
