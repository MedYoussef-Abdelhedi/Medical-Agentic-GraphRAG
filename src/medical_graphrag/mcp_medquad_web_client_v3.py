"""Smoke test for all FOUR MCP tools and the live PubMed search.

  python mcp_medquad_web_client_v3.py --list
  python mcp_medquad_web_client_v3.py --pubmed "Acromegaly"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp_medquad_client_v3 import EXPECTED_TOOLS as OLD_TOOLS, extract_structured

EXPECTED_WEB_TOOLS = OLD_TOOLS | {'rechercher_guidelines_pubmed_actuelles_v1'}


def web_server_parameters():
    from mcp import StdioServerParameters
    base = Path(__file__).resolve().parent
    return StdioServerParameters(
        command=sys.executable,
        args=[str(base / 'mcp_medquad_server_web_v3.py')],
        env={**os.environ, 'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'TOKENIZERS_PARALLELISM': 'false'},
    )


async def cmd(args):
    from mcp import Client
    async with Client(web_server_parameters()) as client:
        tools = await client.list_tools()
        names = {x.name for x in tools.tools}
        if names != EXPECTED_WEB_TOOLS:
            raise RuntimeError(f'Four tools required, received {sorted(names)}')
        print('MCP 4 TOOLS PASS:', ', '.join(sorted(names)))
        if args.list:
            return
        data = extract_structured(await client.call_tool(
            'rechercher_guidelines_pubmed_actuelles_v1', {'topic': args.pubmed, 'max_results': 5}
        ))
        print(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--list', action='store_true')
    g.add_argument('--pubmed', type=str)
    asyncio.run(cmd(p.parse_args()))


if __name__ == '__main__':
    main()
