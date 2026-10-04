"""
Minimal SECOND MCP server — for testing Phase 5 multi-server wiring only.

This is not a real market data feed. It exists purely so you can prove
the backend can connect to and use tools from more than one MCP server
at the same time, before you build (or point at) a real second server.
Swap it out later for whatever that real server ends up being.

Usage:
    cd backend
    python mock_servers/market_data_server.py

This starts an SSE MCP server on port 9001 (the trade server is on 9000,
so the two can run side by side). Then add it to MCP_SERVERS in your
.env, e.g.:

    MCP_SERVERS=[{"name":"trade","transport":"sse","url":"http://localhost:9000/sse"},{"name":"market_data","transport":"sse","url":"http://localhost:9001/sse"}]

Restart the backend and check the startup log line
"Loaded N MCP tool(s): [...]" — it should now include both get_trades /
get_trade_by_id / create_trade (from "trade") AND get_quote (from
"market_data").
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("market-data", port=9001)

# Static, made-up prices — just enough to prove a tool call round-trips
# through a second MCP server correctly.
_MOCK_QUOTES = {
    "AAPL": 227.50,
    "MSFT": 415.00,
    "GOOGL": 178.25,
}


@mcp.tool()
def get_quote(symbol: str) -> dict:
    """
    Returns a mock current price for a stock symbol.

    This is static test data, not a real market feed. It's here only to
    prove that the agent can call a tool from a second MCP server
    alongside the trade server, and that per-server tool namespacing
    works as expected.
    """
    symbol = symbol.upper()
    if symbol not in _MOCK_QUOTES:
        return {"symbol": symbol, "error": "no mock quote for this symbol"}
    return {"symbol": symbol, "price": _MOCK_QUOTES[symbol], "source": "mock_market_data"}


if __name__ == "__main__":
    mcp.run(transport="sse")
