"""Lanceur du prototype, a executer depuis la racine du depot."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "src" / "medical_graphrag"


def main():
    parser = argparse.ArgumentParser(description="Agentic GraphRAG medical")
    parser.add_argument("--mode", choices=("web", "mcp", "hybrid"), default="web")
    args, rest = parser.parse_known_args()
    names = {
        "web": "agent_graphrag_mcp_web_v3.py",
        "mcp": "agent_graphrag_mcp_v3.py",
        "hybrid": "agent_graphrag_hybride_v3_quality.py",
    }
    script = CORE / names[args.mode]
    if not script.is_file():
        parser.error("Script manquant : " + str(script))
    return subprocess.call([sys.executable, str(script), *rest], cwd=str(CORE))


if __name__ == "__main__":
    raise SystemExit(main())
