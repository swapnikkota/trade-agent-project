"""
MCP server registry.

Phase 1: a single trade MCP server, sourced from Settings.trade_mcp_*.
Phase 5: settings.mcp_servers is a config-driven list of N server configs
(set via the MCP_SERVERS env var) so adding a new MCP server is a config
change, not a code change. The Phase 1 trade_mcp_* fields still work as a
fallback when MCP_SERVERS isn't set.

Uses langchain-mcp-adapters' MultiServerMCPClient, which deepagents can
consume directly as a tool source.
"""
from typing import Any, Dict

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import Settings


def _single_server_config_from_legacy_fields(settings: Settings) -> Dict[str, Dict[str, Any]]:
    """
    Phase 1 fallback: builds the same single "trade" server entry as
    before, from the legacy trade_mcp_* settings. Used only when
    settings.mcp_servers is empty.
    """
    if settings.trade_mcp_transport == "http":
        if not settings.trade_mcp_url:
            raise ValueError("TRADE_MCP_URL must be set when TRADE_MCP_TRANSPORT=http")
        # A URL ending in /sse is the legacy SSE transport; streamable_http
        # is the newer transport (typically served at a path like /mcp).
        transport = "sse" if settings.trade_mcp_url.rstrip("/").endswith("/sse") else "streamable_http"
        return {
            "trade": {
                "transport": transport,
                "url": settings.trade_mcp_url,
            }
        }
    elif settings.trade_mcp_transport == "stdio":
        if not settings.trade_mcp_command:
            raise ValueError("TRADE_MCP_COMMAND must be set when TRADE_MCP_TRANSPORT=stdio")
        args = settings.trade_mcp_args.split() if settings.trade_mcp_args else []
        return {
            "trade": {
                "transport": "stdio",
                "command": settings.trade_mcp_command,
                "args": args,
            }
        }
    else:
        raise ValueError(f"Unsupported MCP transport: {settings.trade_mcp_transport}")


def build_mcp_server_configs(settings: Settings) -> Dict[str, Dict[str, Any]]:
    """
    Returns the {server_name: connection_config} dict that
    MultiServerMCPClient expects.

    If settings.mcp_servers is set (Phase 5, config-driven), builds one
    entry per item in that list — this is how you add a second, third,
    etc. MCP server without touching this file. Otherwise falls back to
    the original single "trade" server built from the legacy trade_mcp_*
    fields, so existing .env files keep working unchanged.
    """
    if not settings.mcp_servers:
        return _single_server_config_from_legacy_fields(settings)

    servers: Dict[str, Dict[str, Any]] = {}
    for server in settings.mcp_servers:
        if server.transport in ("sse", "streamable_http"):
            if not server.url:
                raise ValueError(
                    f"MCP server '{server.name}': 'url' is required for transport={server.transport}"
                )
            servers[server.name] = {"transport": server.transport, "url": server.url}
        elif server.transport == "stdio":
            if not server.command:
                raise ValueError(f"MCP server '{server.name}': 'command' is required for transport=stdio")
            servers[server.name] = {
                "transport": "stdio",
                "command": server.command,
                "args": server.args or [],
            }
        else:
            raise ValueError(f"MCP server '{server.name}': unsupported transport {server.transport}")

    return servers


async def get_mcp_tools(settings: Settings):
    """
    Connects to all configured MCP servers and returns the flattened
    list of LangChain-compatible tools deepagents can use. If one server
    in the list is unreachable, MultiServerMCPClient's connection error
    surfaces here and fails the whole startup — by design for now, so a
    misconfigured server is loud rather than silently missing tools.
    """
    configs = build_mcp_server_configs(settings)
    client = MultiServerMCPClient(configs)
    tools = await client.get_tools()
    return tools
