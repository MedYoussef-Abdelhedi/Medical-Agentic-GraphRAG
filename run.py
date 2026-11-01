"""Run an Agentic GraphRAG workflow from the repository root."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CORE = SRC / "medical_graphrag"


def main() -> int:
    parser = argparse.ArgumentParser(description="Medical Agentic GraphRAG")
    parser.add_argument("--mode", choices=("web", "mcp", "hybrid"), default="web")
    args, rest = parser.parse_known_args()
    modules = {
        "web": "medical_graphrag.agents.agent_graphrag_mcp_web_v3",
        "mcp": "medical_graphrag.agents.agent_graphrag_mcp_v3",
        "hybrid": "medical_graphrag.agents.agent_graphrag_hybride_v3_quality",
    }
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.call([sys.executable, "-m", modules[args.mode], *rest], cwd=str(CORE), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
