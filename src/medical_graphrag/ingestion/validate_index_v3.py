"""Independent structural regression check for targeted MedQuAD V3 cleanup.

Run in 2eme_agent: python validate_index_v3.py
No downloads, no LLM calls; reports are written to data/index_v3_quality_report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
EXPECTED_REMOVED = {"8_2": "1", "11_8": "2", "188_10": "February 2014"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(data: Path) -> dict:
    issues: list[str] = []
    paths = {
        "v2_chunks": data / "medquad_chunks_v2.json",
        "v2_vectors": data / "medquad_embeddings_v2.npy",
        "v3_chunks": data / "medquad_chunks_v3.json",
        "v3_vectors": data / "medquad_embeddings_v3.npy",
        "exclusions": data / "index_exclusions_v3.json",
        "manifest": data / "index_manifest_v3.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    m = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    for name, key in [
        ("v2_chunks", "source_chunks_sha256"),
        ("v2_vectors", "source_embeddings_sha256"),
        ("v3_chunks", "chunks_sha256"),
        ("v3_vectors", "embeddings_sha256"),
        ("exclusions", "exclusions_sha256"),
    ]:
        if sha256(paths[name]) != m.get(key):
            issues.append(f"SHA-256 mismatch: {name}")

    v2 = json.loads(paths["v2_chunks"].read_text(encoding="utf-8"))
    v3 = json.loads(paths["v3_chunks"].read_text(encoding="utf-8"))
    exclusions = json.loads(paths["exclusions"].read_text(encoding="utf-8"))
    vectors2 = np.load(paths["v2_vectors"], allow_pickle=False)
    vectors3 = np.load(paths["v3_vectors"], allow_pickle=False)

    if len(v2) != 753 or len(v3) != 750:
        issues.append(f"Incorrect chunk counts: {len(v2)} -> {len(v3)}")
    if vectors2.shape != (len(v2), 384) or vectors3.shape != (len(v3), 384):
        issues.append(f"Incorrect vector dimensions: {vectors2.shape}, {vectors3.shape}")
    if not np.isfinite(vectors3).all():
        issues.append("Non-finite V3 embeddings")
    if not np.allclose(np.linalg.norm(vectors3, axis=1), 1, atol=1e-3, rtol=0):
        issues.append("V3 vectors not normalized")

    original_lookup = {c["chunk_id"]: (i, c) for i, c in enumerate(v2)}
    ids3 = [c["chunk_id"] for c in v3]
    if len(ids3) != len(set(ids3)):
        issues.append("Duplicate V3 chunk IDs")
    if set(ids3) != set(original_lookup) - set(EXPECTED_REMOVED):
        issues.append("Unexpected removed or added chunks")
    if {c["document_id"] for c in v2} != {c["document_id"] for c in v3}:
        issues.append("Document was added/lost")

    retained_positions = []
    for row in v3:
        cid = row["chunk_id"]
        if cid not in original_lookup:
            continue
        pos, original = original_lookup[cid]
        retained_positions.append(pos)
        expected = dict(original)
        if int(row["document_id"]) == 188:
            expected["source_date_text"] = "February 2014"
            expected["source_date_evidence_chunk_id"] = "188_10"
        if row != expected:
            issues.append(f"Changed text, metadata or record: {cid}")
        if len(row["content"].strip()) <= 20:
            issues.append(f"Short fragment was not cleaned: {cid}")
    if len(retained_positions) == len(v3) and vectors3.shape == (len(v3), 384):
        if not np.array_equal(vectors3, vectors2[retained_positions]):
            issues.append("V3 vector rows differ from corresponding V2 rows")

    logged = exclusions.get("excluded_chunks", [])
    if {item["original_chunk"]["chunk_id"] for item in logged} != set(EXPECTED_REMOVED):
        issues.append("Missing or unexpected exclusion-log entries")
    for item in logged:
        original = item["original_chunk"]
        cid = original["chunk_id"]
        if cid in EXPECTED_REMOVED and original["content"] != EXPECTED_REMOVED[cid]:
            issues.append(f"Exclusion text mismatch: {cid}")
        if cid in original_lookup and original != original_lookup[cid][1]:
            issues.append(f"Exclusion audit trail mismatch: {cid}")

    report = {
        "status": "FAIL" if issues else "PASS_TECHNICAL_WITH_REVIEW",
        "v2_chunks": len(v2),
        "v3_chunks": len(v3),
        "documents": len({c["document_id"] for c in v3}),
        "v3_embeddings_shape": list(vectors3.shape),
        "excluded_fragments": sorted(EXPECTED_REMOVED),
        "source_date_metadata_preserved": all(
            c.get("source_date_text") == "February 2014" and c.get("source_date_evidence_chunk_id") == "188_10"
            for c in v3 if int(c["document_id"]) == 188
        ),
        "issues": issues,
        "limitations": [
            "V2 embedding correspondence relies on V2 row alignment (no original embedding hash in V2 manifest).",
            "Source date is retained as literal evidence; not independently certified publication date.",
            "Historical clinical data not validated for medical accuracy or freshness.",
            "Independent retrieval benchmark and RAGAS evaluation not yet performed.",
        ],
    }
    if not report["source_date_metadata_preserved"]:
        issues.append("Source date metadata not preserved")
        report["status"] = "FAIL"
    (data / "index_v3_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    args = parser.parse_args()
    report = validate(args.data_dir)
    print("STATUS:", report["status"])
    print("V2 -> V3 chunks:", report["v2_chunks"], "->", report["v3_chunks"])
    print("Documents:", report["documents"])
    print("Embeddings:", tuple(report["v3_embeddings_shape"]))
    print("Excluded:", report["excluded_fragments"])
    print("Source date preserved:", report["source_date_metadata_preserved"])
    print("Errors:", report["issues"])
    print("Report:", args.data_dir / "index_v3_quality_report.json")
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
