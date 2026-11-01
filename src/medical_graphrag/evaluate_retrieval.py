"""Benchmark exploratoire de recherche sur MedQuAD: comparaison V1, V2, V3.

Usage (from 2eme_agent in an activated .venv):
    python evaluate_retrieval.py

No Groq calls, no file mutation in the existing indices.
This uses SOURCE-ANCHORED relevance labels, not externally validated medical labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
BENCHMARK = BASE / "benchmark_candidates_v1.json"
OUT = BASE / "reports" / "retrieval_benchmark_v1"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_K = 5
FILES = {
    "V1": ("medquad_chunks.json", "medquad_embeddings.npy"),
    "V2": ("medquad_chunks_v2.json", "medquad_embeddings_v2.npy"),
    "V3": ("medquad_chunks_v3.json", "medquad_embeddings_v3.npy"),
}


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def get_gold_ids(ex: dict, chunks: list[dict]) -> set[str]:
    """Match a verified evidence phrase to the *same source document*.

    Not exhaustive relevance judgments: other chunks may also be relevant.
    """
    target = normalized(ex["reference_evidence_phrase"])
    doc_id = int(ex["reference_document_id"])
    return {
        row["chunk_id"]
        for row in chunks
        if int(row["document_id"]) == doc_id
        and target in normalized(row["content"])
    }


def first_gold_rank(sorted_positions: np.ndarray, chunks: list[dict], gold_ids: set[str]) -> int | None:
    for rank, idx in enumerate(sorted_positions, 1):
        if chunks[int(idx)]["chunk_id"] in gold_ids:
            return rank
    return None


def metric_summary(ranks: list[int | None]) -> dict:
    if not ranks:
        raise ValueError("No answerable queries available")
    n = len(ranks)
    return {
        "query_count": n,
        "AnchorHit@1": sum(r == 1 for r in ranks) / n,
        "AnchorHit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "AnchorHit@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "AnchorMRR@5": sum((1.0 / r) if r is not None and r <= 5 else 0 for r in ranks) / n,
    }


def load_index(version: str) -> tuple[list[dict], np.ndarray, dict]:
    filename, vecname = FILES[version]
    chunk_path, matrix_path = DATA / filename, DATA / vecname
    for path in (chunk_path, matrix_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; build and validate that index first.")
    chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
    vectors = np.load(matrix_path, allow_pickle=False)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"{version}: empty or invalid chunk list")
    if vectors.shape != (len(chunks), 384):
        raise ValueError(f"{version}: vector shape {vectors.shape}, expected ({len(chunks)}, 384)")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{version}: non-finite embeddings")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1, atol=1e-3, rtol=0):
        raise ValueError(f"{version}: vectors are not L2 normalized")
    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{version}: duplicated chunk IDs")
    return chunks, vectors.astype(np.float32, copy=False), {
        "chunks_file": filename,
        "chunks_sha256": sha256_file(chunk_path),
        "embeddings_file": vecname,
        "embeddings_sha256": sha256_file(matrix_path),
        "chunks_count": len(chunks),
        "embedding_dimensions": vectors.shape[1],
    }


def evaluate_version(
    version: str, chunks: list[dict], vectors: np.ndarray,
    examples: list[dict], query_vectors: np.ndarray,
) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    for ex, qvec in zip(examples, query_vectors):
        sims = vectors @ qvec
        positions = np.argsort(sims)[::-1][:MAX_K]
        evidence_ids: set[str] = set()
        gold_rank = None
        if ex["type"] == "answerable":
            evidence_ids = get_gold_ids(ex, chunks)
            if not evidence_ids:
                raise ValueError(
                    f"{version} {ex['id']}: evidence anchor not found in doc "
                    f"{ex['reference_document_id']} (data changed or incomplete label)"
                )
            gold_rank = first_gold_rank(positions, chunks, evidence_ids)

        top_results = [
            {
                "rank": rank,
                "chunk_id": chunks[int(pos)]["chunk_id"],
                "document_id": int(chunks[int(pos)]["document_id"]),
                "score": float(sims[int(pos)]),
                "is_anchor": chunks[int(pos)]["chunk_id"] in evidence_ids,
                "excerpt": " ".join(chunks[int(pos)]["content"].split())[:200],
            }
            for rank, pos in enumerate(positions, 1)
        ]
        rows.append({
            "id": ex["id"], "type": ex["type"], "domain": ex["domain"],
            "query": ex["query"], "version": version,
            "gold_rank_top5": gold_rank,
            "gold_chunk_ids": sorted(evidence_ids),
            "top5": top_results,
        })
    scores = metric_summary([
        r["gold_rank_top5"] for r in rows if r["type"] == "answerable"
    ])
    return scores, rows


def bootstrap_hit3_delta(by_version: dict[str, list[dict]], a: str, b: str) -> dict:
    """Paired bootstrap interval for Hit@3(b) - Hit@3(a), descriptive only."""
    ra = {r["id"]: r["gold_rank_top5"] for r in by_version[a] if r["type"] == "answerable"}
    rb = {r["id"]: r["gold_rank_top5"] for r in by_version[b] if r["type"] == "answerable"}
    if set(ra) != set(rb):
        raise ValueError("Queries do not match across versions")
    deltas = np.array([
        int(rb[k] is not None and rb[k] <= 3) - int(ra[k] is not None and ra[k] <= 3)
        for k in sorted(ra)
    ], dtype=float)
    generator = np.random.default_rng(42)
    boot = np.mean(deltas[generator.integers(0, len(deltas), (2000, len(deltas)))], axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {"comparison": f"{b} minus {a}", "difference": float(np.mean(deltas)),
            "bootstrap_95pct_CI": [float(low), float(high)], "resamples": 2000}


def main() -> None:
    if not BENCHMARK.is_file():
        raise FileNotFoundError(f"Missing {BENCHMARK}; put it beside this script.")
    bench = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    examples = bench["examples"]
    ids = [x["id"] for x in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicated benchmark IDs")
    if len([e for e in examples if e["type"] == "answerable"]) != bench["positives"]:
        raise ValueError("Unexpected positive count in benchmark manifest")
    if len([e for e in examples if e["type"] == "unsupported_probe"]) != bench["unsupported_probes"]:
        raise ValueError("Unexpected unsupported probe count")
    if any(not e["query"].strip() for e in examples):
        raise ValueError("Empty query")

    indexes = {version: load_index(version) for version in FILES}
    print("Benchmark:", len(examples), "queries, including", bench["positives"], "with evidence anchors")
    print("Loading MiniLM for query embeddings (no Groq calls)...")
    from langchain_huggingface import HuggingFaceEmbeddings

    model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    query_vectors = np.asarray(model.embed_documents([e["query"] for e in examples]), dtype=np.float32)
    if query_vectors.shape != (len(examples), 384):
        raise ValueError(f"Unexpected query embedding shape {query_vectors.shape}")
    if not np.allclose(np.linalg.norm(query_vectors, axis=1), 1.0, atol=1e-3, rtol=0):
        raise ValueError("Query embeddings not normalized")

    all_rows, summaries, index_hashes = {}, {}, {}
    for version, (chunks, vectors, metadata) in indexes.items():
        stats, details = evaluate_version(version, chunks, vectors, examples, query_vectors)
        all_rows[version], summaries[version], index_hashes[version] = details, stats, metadata
        print(f"{version} | chunks={len(chunks)} | AnchorHit@1={stats['AnchorHit@1']:.1%} "
              f"| AnchorHit@3={stats['AnchorHit@3']:.1%} "
              f"| AnchorHit@5={stats['AnchorHit@5']:.1%} "
              f"| AnchorMRR@5={stats['AnchorMRR@5']:.4f}")

    comparisons = [bootstrap_hit3_delta(all_rows, "V1", "V2"),
                   bootstrap_hit3_delta(all_rows, "V2", "V3"),
                   bootstrap_hit3_delta(all_rows, "V1", "V3")]
    OUT.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_file": BENCHMARK.name,
        "benchmark_sha256": sha256_file(BENCHMARK),
        "benchmark_status": bench["benchmark_status"],
        "embedding_model": MODEL_NAME,
        "top_k": MAX_K,
        "metrics_scope": "anchor-evidence retrieval, NOT full relevance or medical quality",
        "index_hashes": index_hashes,
        "summaries": summaries,
        "paired_hit3_differences": comparisons,
        "details": all_rows,
    }
    (OUT / "results.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "per_query.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["version", "id", "type", "domain", "query", "gold_rank_top5", "gold_chunk_ids", "top_1_chunk", "top_1_score"])
        writer.writeheader()
        for version, rows in all_rows.items():
            for item in rows:
                writer.writerow({
                    "version": version, "id": item["id"], "type": item["type"],
                    "domain": item["domain"], "query": item["query"],
                    "gold_rank_top5": item["gold_rank_top5"] if item["gold_rank_top5"] is not None else "",
                    "gold_chunk_ids": ";".join(item["gold_chunk_ids"]),
                    "top_1_chunk": item["top5"][0]["chunk_id"],
                    "top_1_score": round(item["top5"][0]["score"], 6),
                })
    lines = [
        "# Retrieval comparison — exploratory MedQuAD benchmark", "",
        "**IMPORTANT:** Source-anchored candidate benchmark, not independently adjudicated or clinically validated.",
        "Medical information is historical. This benchmark does not evaluate the LLM answer.", "",
        "| Index | Chunks | AnchorHit@1 | AnchorHit@3 | AnchorHit@5 | AnchorMRR@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for version, stat in summaries.items():
        lines.append(f"| {version} | {index_hashes[version]['chunks_count']} | {stat['AnchorHit@1']:.1%} | {stat['AnchorHit@3']:.1%} | {stat['AnchorHit@5']:.1%} | {stat['AnchorMRR@5']:.4f} |")
    lines += ["", "## Paired Hit@3 differences (indicative bootstrap)", ""]
    for comp in comparisons:
        lo, hi = comp["bootstrap_95pct_CI"]
        lines.append(f"- {comp['comparison']}: {comp['difference']:+.1%} (95% bootstrap CI: {lo:+.1%} to {hi:+.1%}).")
    lines += ["", "## Queries where reference evidence was absent from Top-5", ""]
    for version, rows in all_rows.items():
        missing = [x for x in rows if x["type"] == "answerable" and x["gold_rank_top5"] is None]
        lines.append(f"### {version}: {len(missing)} queries")
        for row in missing:
            lines.append(f"- {row['id']}: {row['query']} (top1={row['top5'][0]['chunk_id']}; reference evidence={','.join(row['gold_chunk_ids'])})")
    lines += ["", "## Limits and review", "",
              "- Questions were paraphrased manually but the author read source passages: this is **not blind external evaluation**.",
              "- Evidence-anchor labels are not exhaustive relevance judgments: retrieval of an alternate valid passage may be scored as a miss.",
              "- The five unsupported probes are **not** included in retrieval metrics. Their results are only logged for later abstention tests.",
              "- The queries and labels should be checked by an independent reviewer and frozen before tuning or comparing new models.",
              "- No clinical validity, guideline freshness, or LLM faithfulness is evaluated here.",
              "- Keep the exact indices and their recorded checksums for reproducibility.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nWrote:", OUT / "summary.md")
    print("       ", OUT / "per_query.csv")
    print("       ", OUT / "results.json")
    print("Review the misses, duplicate-document hits and unsupported-probe traces before choosing V3.")


if __name__ == "__main__":
    main()
