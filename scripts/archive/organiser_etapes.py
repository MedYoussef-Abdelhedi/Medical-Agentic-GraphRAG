"""Classe les scripts d'un projet MedQuAD en copies par étape, sans casser les imports.

Usage (depuis 2eme_agent):
  python organiser_etapes.py --plan
  python organiser_etapes.py --apply
  python organiser_etapes.py --verify

Ne déplace, ne supprime et ne modifie AUCUN fichier source, .env, data ou reports.
N'utilise que la bibliothèque standard Python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

STAGES = {
    "00_environnement": "Snapshots de dépendances et configuration non secrète.",
    "01_preparation_corpus": "Préparation et nettoyage de MedQuAD.",
    "02_indexation_v3": "Construction, validation et audit des index.",
    "03_evaluation_retrieval": "Recherche sémantique et benchmark du retrieval.",
    "04_evaluation_ragas": "Évaluation RAGAS exploratoire.",
    "05_agent_langgraph": "Agents LangGraph initiaux et leurs tests.",
    "06_graphe_neo4j": "Import et pilote clinique documentaire Neo4j.",
    "07_graphrag_hybride": "Première version hybride Vector + Neo4j.",
    "08_comparaison_qualite": "Comparaison, prompts de qualité et tests de régression.",
    "09_integration_mcp": "MCP historique : serveur, client, agent et tests.",
    "10_mcp_pubmed": "MCP PubMed : web actuel, citations et tests.",
    "99_a_classer": "Scripts non reconnus : classement manuel conseillé.",
}

SUPPORTED_SUFFIXES = {".py", ".json", ".md", ".txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_for(filename: str) -> str:
    n = filename.lower()
    # Les scripts PubMed/web MCP doivent primer sur les règles MCP générales.
    if ("pubmed" in n or "mcp_web" in n or "mcp_medquad_web" in n
        or "agent_graphrag_mcp_web" in n or "readme_etape_16b" in n):
        return "10_mcp_pubmed"
    if ("mcp" in n or n.startswith("readme_etape_16_")):
        return "09_integration_mcp"
    if ("quality" in n and ("agent_graphrag" in n or "generation" in n or "compare" in n)
        or "compare_vector_graphrag" in n
        or "readme_etape_15" in n
        or "evaluation_qualitative" in n):
        return "08_comparaison_qualite"
    if ("graphrag_hybride" in n or "readme_etape_14" in n):
        return "07_graphrag_hybride"
    if ("clinical_graph" in n or "import_medquad" in n or "neo4j" in n
        or "readme_etape_13" in n):
        return "06_graphe_neo4j"
    if ("langgraph" in n or "readme_etape_12" in n):
        return "05_agent_langgraph"
    if ("ragas" in n or "rag_eval" in n or "readme_etape_11" in n):
        return "04_evaluation_ragas"
    if ("retrieval" in n or "similarity_search" in n or "benchmark" in n
        or "evaluation_retrieval" in n):
        return "03_evaluation_retrieval"
    if ("index" in n or "audit_pipeline" in n or "embedding" in n):
        return "02_indexation_v3"
    if ("chunk" in n or "corpus" in n or n == "check_quality.py" or "medquad" in n):
        return "01_preparation_corpus"
    if n.startswith("requirements") or n in {"pyproject.toml", "pip-freeze.txt"}:
        return "00_environnement"
    return "99_a_classer"


def source_files(root: Path) -> list[Path]:
    items = []
    for p in root.iterdir():
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if p.name in {"organiser_etapes.py", "manifest_etapes.json", "README_ORGANISATION.md"}:
            continue
        # Ne jamais copier fichier secret, état personnel ou scripts de sauvegarde en dehors des étapes.
        if p.name.startswith("."):
            continue
        items.append(p)
    return sorted(items, key=lambda x: x.name.lower())


def manifest_path(root: Path) -> Path:
    return root / "etapes" / "manifest_etapes.json"


def stage_destination(root: Path, src: Path) -> Path:
    stage = stage_for(src.name)
    if "backup" in src.stem.lower() or "bak" == src.suffix.lower():
        return root / "etapes" / stage / "versions_anterieures" / src.name
    return root / "etapes" / stage / src.name


def print_plan(root: Path, sources: list[Path]) -> None:
    grouped: dict[str, list[str]] = {s: [] for s in STAGES}
    for src in sources:
        dest = stage_destination(root, src)
        grouped[dest.relative_to(root / "etapes").parts[0]].append(src.name)
    print(f"PROJET : {root}")
    print("MODE : PLAN seulement — aucun fichier modifié.\n")
    for stage, files in grouped.items():
        if files:
            print(f"{stage} ({len(files)})")
            for f in files:
                print(f"  - {f}")
    print(f"\nTOTAL : {len(sources)} fichiers classés en copies.")
    print("La racine, .env, data, reports et .venv restent intacts.")


def apply(root: Path, sources: list[Path]) -> int:
    target_root = root / "etapes"
    conflicts = []
    for src in sources:
        dest = stage_destination(root, src)
        if dest.exists() and sha256(src) != sha256(dest):
            conflicts.append(dest)
    if conflicts:
        print("ABANDON : des copies existent et ont un contenu différent.")
        for c in conflicts:
            print(" -", c.relative_to(root))
        print("Aucun fichier n'a été écrasé. Sauvegarde/compare les versions avant de recommencer.")
        return 2

    target_root.mkdir(exist_ok=True)
    copied, unchanged = 0, 0
    entries = []
    for src in sources:
        dest = stage_destination(root, src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
            copied += 1
        else:
            unchanged += 1
        entries.append({
            "source": src.name,
            "copy": str(dest.relative_to(root)).replace("\\", "/"),
            "sha256": sha256(src),
        })

    for stage, description in STAGES.items():
        stage_dir = target_root / stage
        if not stage_dir.is_dir():
            continue
        readme = stage_dir / "README_ORGANISATION.md"
        if not readme.exists():
            readme.write_text(
                f"# {stage}\n\n{description}\n\n"
                "Ces fichiers sont des **copies de classement**, pas encore un package Python refactorisé.\n"
                "Lance les scripts originaux depuis la racine `2eme_agent` jusqu'à la migration des imports.\n",
                encoding="utf-8",
            )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "non_destructive_copy",
        "source_project": str(root),
        "files": entries,
    }
    manifest_path(root).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"APPLY PASS : {copied} copies créées ; {unchanged} déjà identiques ; {len(entries)} dans le manifeste.")
    print("Sources intactes. Le dossier `etapes` est un classement, PAS encore la nouvelle racine d'exécution.")
    return 0


def verify(root: Path) -> int:
    path = manifest_path(root)
    if not path.is_file():
        print("VERIFY FAIL : exécute d'abord --apply")
        return 2
    content = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    for record in content.get("files", []):
        original, copied = root / record["source"], root / record["copy"]
        for label, f in [("original", original), ("copie", copied)]:
            if not f.is_file():
                errors.append(f"{label} introuvable : {f}")
            elif sha256(f) != record["sha256"]:
                errors.append(f"{label} différent du manifeste : {f}")
    if errors:
        print("VERIFY FAIL :")
        for err in errors:
            print(" -", err)
        return 2
    print(f"VERIFY PASS : {len(content['files'])} fichiers source et copies identiques (SHA256).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Organiser MedQuAD en étapes sans casser les imports")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true")
    group.add_argument("--apply", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not (root / "data").is_dir():
        print("ERREUR : placer organiser_etapes.py à la racine de 2eme_agent, à côté de data/.")
        return 2
    if args.verify:
        return verify(root)
    sources = source_files(root)
    if args.plan:
        print_plan(root, sources)
        return 0
    return apply(root, sources)


if __name__ == "__main__":
    sys.exit(main())
