"""Generate a candidate requirements.txt from project imports and the active Python env.

Run from the repository root with:
    python scripts/generate_requirements.py
    python scripts/generate_requirements.py --write

This is NOT a clean-install validation or a complete reproducible lockfile.
Dynamic imports, optional features and platform-specific package builds need review.
"""

from __future__ import annotations

import argparse
import ast
from importlib import metadata
from pathlib import Path
import re
import sys

# Import module -> PyPI distribution, where the names differ or discovery is ambiguous.
MODULE_TO_DIST = {
    "dotenv": "python-dotenv",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "sentence_transformers": "sentence-transformers",
    "langchain_groq": "langchain-groq",
    "langchain_core": "langchain-core",
    "langchain_community": "langchain-community",
    "langchain_huggingface": "langchain-huggingface",
    "langchain_text_splitters": "langchain-text-splitters",
    "langchain_neo4j": "langchain-neo4j",
    "langgraph": "langgraph",
    "langchain": "langchain",
    "neo4j": "neo4j",
    "groq": "groq",
    "mcp": "mcp",
    "fastmcp": "fastmcp",
    "ragas": "ragas",
    "torch": "torch",
    "numpy": "numpy",
    "pandas": "pandas",
    "datasets": "datasets",
    "transformers": "transformers",
    "huggingface_hub": "huggingface-hub",
    "tiktoken": "tiktoken",
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "googleapiclient": "google-api-python-client",
    "pytest": "pytest",
    "httpx": "httpx",
    "requests": "requests",
    "scipy": "scipy",
    "faiss": "faiss-cpu",  # Check this manually if your environment uses faiss-gpu.
    "rank_bm25": "rank-bm25",
    "openai": "openai",
    "pydantic": "pydantic",
    "tenacity": "tenacity",
    "anyio": "anyio",
    "aiohttp": "aiohttp",
    "tqdm": "tqdm",
}


def source_files(root: Path) -> list[Path]:
    selected = [root / "run.py", root / "run_tests.py"]
    for folder in (root / "src", root / "tests"):
        if folder.is_dir():
            selected.extend(sorted(folder.rglob("*.py")))
    return [path for path in selected if path.is_file()]


def collect_imports(files: list[Path]) -> tuple[set[str], list[str]]:
    modules: set[str] = set()
    errors: list[str] = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.partition(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.partition(".")[0])
    return modules, errors


def normalize_distribution(s: str) -> str:
    return re.sub(r"[-_.]+", "-", s).lower()


def generate(root: Path) -> tuple[str, list[str], list[str]]:
    files = source_files(root)
    if not files:
        raise RuntimeError("No Python sources found in src/, tests/, run.py or run_tests.py")
    imports, parse_errors = collect_imports(files)
    if parse_errors:
        raise RuntimeError("Cannot parse:\n" + "\n".join(parse_errors))

    local_modules = {p.stem for p in files}
    local_modules.update({"src", "tests", "medical_graphrag"})
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
    installed_map = metadata.packages_distributions()
    selected: dict[str, str] = {}
    unresolved: list[str] = []
    ambiguous: list[str] = []

    for module in sorted(imports - local_modules - stdlib):
        preferred = MODULE_TO_DIST.get(module)
        candidates = [preferred] if preferred else installed_map.get(module, [])
        if not candidates:
            unresolved.append(module)
            continue
        if not preferred and len(candidates) != 1:
            ambiguous.append(f"{module}: {', '.join(candidates)}")
            continue
        package = candidates[0]
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            unresolved.append(f"{module} (distribution {package} missing)")
            continue
        # Do not commit a CUDA/local-build pin that may be unresolvable from PyPI.
        if "+" in version:
            ambiguous.append(f"{module}: installed {package}=={version} is a local/platform-specific build")
            version = version.split("+", 1)[0]
        selected[normalize_distribution(package)] = f"{package}=={version}"

    lines = [
        "# Candidate direct dependencies derived from Python imports in src/, tests/ and run*.py.",
        "# Versions reflect the CURRENT active Python environment; independently validate a clean install.",
        "# Python 3.12 was used during local project development.",
        "# Generated by scripts/generate_requirements.py; review unresolved/dynamic imports.",
        "",
    ]
    lines.extend(selected[k] for k in sorted(selected))
    lines.append("")
    return "\n".join(lines), unresolved, ambiguous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Save candidate requirements.txt at repository root")
    parser.add_argument("--root", type=Path, help="Override repository root")
    args = parser.parse_args()
    root = (args.root or Path(__file__).resolve().parent.parent).resolve()
    print(f"Repository: {root}")
    print(f"Interpreter: {sys.executable}")
    print(f"Python: {sys.version.split()[0]}")
    try:
        result, unresolved, ambiguous = generate(root)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("\n--- PROPOSED requirements.txt ---\n" + result)
    if unresolved:
        print("REVIEW - imports without an installed matching distribution:")
        for item in unresolved:
            print("  ?", item)
    if ambiguous:
        print("REVIEW - ambiguous modules or local builds:")
        for item in ambiguous:
            print("  ?", item)
    if args.write:
        output = root / "requirements.txt"
        output.write_text(result, encoding="utf-8", newline="\n")
        print(f"\nCREATED: {output}")
    print("\nNEXT: review warnings, run 'python -m pip check', test a FRESH venv,")
    print("then run 'python run_tests.py' and 'python run.py --check' before committing.")
    if unresolved or ambiguous:
        print("WARNING: review detected issues before calling the file reproducible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
