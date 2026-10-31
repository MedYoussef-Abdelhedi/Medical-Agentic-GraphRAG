"""Actual MCP stdio client for testing MedQuAD V3 server without Groq.

    python mcp_medquad_client_v3.py --list
    python mcp_medquad_client_v3.py --stats
    python mcp_medquad_client_v3.py --graph "Acromegaly"
    python mcp_medquad_client_v3.py --search "What are the symptoms of acromegaly?"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SERVER = BASE / "mcp_medquad_server_v3.py"
EXPECTED_TOOLS = {"statistiques_medquad_v3", "rechercher_passages_v3", "symptomes_neo4j_v3"}


def server_parameters():
    from mcp import StdioServerParameters

    return StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"},
    )


def _payload_from_json(value, *, max_unwrap: int = 3) -> dict | None:
    """Accept a JSON object, or a single SDK {"result": ...} envelope.

    Never use eval/ast.literal_eval or parse arbitrary server prose. Only a
    recognized status-bearing JSON object is accepted as an MCP tool payload.
    """
    for _ in range(max_unwrap):
        if isinstance(value, dict):
            if isinstance(value.get("status"), str):
                if value["status"] not in {"ok", "not_supported"}:
                    raise ValueError(f"Statut MCP inattendu: {value['status']!r}")
                return value
            if set(value) == {"result"}:
                value = value["result"]
                continue
            return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                return None
            continue
        return None
    return None


def extract_structured(result) -> dict:
    """Read MCP 2 results without guessing or silently trusting arbitrary text.

    The official SDK normally returns `structured_content` for typed tools.
    Some versions/transport adapters wrap it in {"result": ...} or only send
    JSON in a TextContent block. Accept those narrowly and validate status.
    """
    if getattr(result, "is_error", False):
        details = "; ".join(str(getattr(x, "text", x)) for x in getattr(result, "content", []))
        raise RuntimeError(f"MCP tool failed: {details[:400]}")

    structured = getattr(result, "structured_content", None)
    data = _payload_from_json(structured)

    blocks = getattr(result, "content", []) or []
    text_payload = None
    if len(blocks) == 1 and getattr(blocks[0], "type", None) == "text":
        text_payload = _payload_from_json(getattr(blocks[0], "text", None))

    if data is not None and text_payload is not None and data != text_payload:
        raise ValueError("Incohérence entre structured_content et content MCP")
    if data is not None:
        return data
    if text_payload is not None:
        return text_payload

    shape = type(structured).__name__
    keys = sorted(str(k) for k in structured.keys()) if isinstance(structured, dict) else []
    content_types = [str(getattr(b, "type", type(b).__name__)) for b in blocks]
    raise ValueError(
        f"Réponse MCP non reconnue : structured_content={shape}, "
        f"clés={keys[:12]}, blocs={content_types[:6]}. "
        "Aucune donnée d'outil n'a été acceptée."
    )


async def mcp_cli(args) -> None:
    from mcp import Client

    async with Client(server_parameters()) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}
        if names != EXPECTED_TOOLS:
            raise RuntimeError(f"Outils MCP inattendus: {sorted(names)}")
        print("HANDSHAKE MCP: PASS")
        if args.list:
            print("OUTILS MCP:", ", ".join(sorted(names)))
            return
        if args.stats:
            result = await client.call_tool("statistiques_medquad_v3", {})
        elif args.graph is not None:
            result = await client.call_tool("symptomes_neo4j_v3", {"disease": args.graph})
        else:
            result = await client.call_tool("rechercher_passages_v3", {"question": args.search, "top_k": 3})
        print(json.dumps(extract_structured(result), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--stats", action="store_true")
    group.add_argument("--graph", type=str)
    group.add_argument("--search", type=str)
    args = parser.parse_args()
    asyncio.run(mcp_cli(args))


if __name__ == "__main__":
    main()
