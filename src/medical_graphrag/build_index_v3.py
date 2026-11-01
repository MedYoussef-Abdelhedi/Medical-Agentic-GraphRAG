"""MedQuAD V2 -> V3: targeted, audited removal from semantic index.

Run from 2eme_agent, using its venv:
    python build_index_v3.py

No LLM calls or downloads. Original V1/V2 inputs are never modified.
This is a TECHNICAL index cleanup, not a medical-content certification.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent

# Exact decisions from human review of the uploaded V2 JSON, not a broad
# rule such as "delete all short chunks".
DECISIONS = {
    "8_2": {
        "document_id": 8,
        "source_file": "0000002.xml",
        "content": "1",
        "reason": "Isolated digit, likely a bibliography/footnote residue; not useful as a standalone medical passage.",
    },
    "11_8": {
        "document_id": 11,
        "source_file": "0000002.xml",
        "content": "2",
        "reason": "Isolated digit, likely a bibliography/footnote residue; not useful as a standalone medical passage.",
    },
    "188_10": {
        "document_id": 188,
        "source_file": "0000048.xml",
        "content": "February 2014",
        "reason": "Standalone date; preserved as source_date_text metadata for document 188.",
    },
}
SOURCE_DATE_DOC_ID = 188
SOURCE_DATE_TEXT = "February 2014"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(data: object) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def store_atomic(path: Path, payload: bytes) -> None:
    """Write via a temporary file; publish each completed file atomically."""
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    try:
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def build(data: Path, overwrite: bool = False) -> dict:
    in_chunks = data / "medquad_chunks_v2.json"
    in_vectors = data / "medquad_embeddings_v2.npy"
    in_manifest = data / "index_manifest_v2.json"
    output = {
        "chunks": data / "medquad_chunks_v3.json",
        "vectors": data / "medquad_embeddings_v3.npy",
        "exclusions": data / "index_exclusions_v3.json",
        "manifest": data / "index_manifest_v3.json",
    }
    for input_path in (in_chunks, in_vectors, in_manifest):
        if not input_path.is_file():
            raise FileNotFoundError(f"Required V2 file is missing: {input_path}")
    collisions = [str(p) for p in output.values() if p.exists()]
    if collisions and not overwrite:
        raise FileExistsError("V3 already exists. No files were changed: " + ", ".join(collisions))

    old_manifest = json.loads(in_manifest.read_text(encoding="utf-8"))
    if sha256_file(in_chunks) != old_manifest.get("chunks_sha256"):
        raise ValueError("V2 JSON SHA-256 does not match its manifest; stop and investigate.")
    old = json.loads(in_chunks.read_text(encoding="utf-8"))
    vectors = np.load(in_vectors, allow_pickle=False)
    if not isinstance(old, list) or len(old) != 753:
        raise ValueError(f"Expected the validated 753-chunk V2 corpus, found {len(old)}.")
    if old_manifest.get("chunks_after") != len(old):
        raise ValueError("V2 manifest chunk count mismatch.")
    if vectors.shape != (len(old), 384) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid V2 embeddings: {vectors.shape}.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3, rtol=0):
        raise ValueError("V2 vectors must be L2-normalized.")
    source_count = old_manifest.get("max_input_tokens_observed")
    if not isinstance(source_count, int) or source_count > 256:
        raise ValueError("V2 token budget not attested by its manifest.")

    ids = [c["chunk_id"] for c in old]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs in V2.")
    found = set(ids) & set(DECISIONS)
    if found != set(DECISIONS):
        raise ValueError(f"Expected chunks not found in V2: {sorted(set(DECISIONS)-found)}")
    other_short = [c["chunk_id"] for c in old
                   if len(c["content"].strip()) <= 20 and c["chunk_id"] not in DECISIONS]
    if other_short:
        raise ValueError(f"Unexpected short passages require separate review: {other_short}")

    selected_positions: list[int] = []
    cleaned: list[dict] = []
    exclusions: list[dict] = []
    for i, chunk in enumerate(old):
        cid = chunk["chunk_id"]
        if chunk["text_for_embedding"] != f"Question: {chunk['question']}\nAnswer: {chunk['content']}":
            raise ValueError(f"Invalid text_for_embedding for {cid}.")
        if cid in DECISIONS:
            rule = DECISIONS[cid]
            if (chunk["document_id"] != rule["document_id"]
                    or chunk["source_file"] != rule["source_file"]
                    or chunk["content"] != rule["content"]):
                raise ValueError(f"Content/source of {cid} has changed; refuse to exclude.")
            exclusions.append({
                "old_row_index": i,
                "review_decision": "exclude_from_semantic_index_only",
                "reason": rule["reason"],
                "original_chunk": chunk,
            })
            continue
        new_chunk = dict(chunk)  # preserve the ORIGINAL chunk_id and embedding text
        if int(new_chunk["document_id"]) == SOURCE_DATE_DOC_ID:
            new_chunk["source_date_text"] = SOURCE_DATE_TEXT
            new_chunk["source_date_evidence_chunk_id"] = "188_10"
        cleaned.append(new_chunk)
        selected_positions.append(i)

    new_vectors = vectors[np.asarray(selected_positions, dtype=np.int64)].copy()
    if len(cleaned) != 750 or new_vectors.shape != (750, 384):
        raise AssertionError("Unexpected V3 output size.")
    if {c["document_id"] for c in old} != {c["document_id"] for c in cleaned}:
        raise AssertionError("Some source documents disappeared.")
    if any(len(c["content"].strip()) <= 20 for c in cleaned):
        raise AssertionError("Short fragments remain in V3.")
    if not all(np.array_equal(new_vectors[j], vectors[i])
               for j, i in enumerate(selected_positions)):
        raise AssertionError("A retained embedding was altered.")

    exclusions_data = {
        "policy": "Specific manual decisions; V1/V2 and the full excluded payloads are preserved.",
        "excluded_chunks": exclusions,
        "metadata_preserved": {
            "document_id": SOURCE_DATE_DOC_ID,
            "field": "source_date_text",
            "value": SOURCE_DATE_TEXT,
            "evidence_original_chunk_id": "188_10",
            "interpretation_warning": "Date appears in source; publication-date status has not been independently confirmed.",
        },
    }
    chunk_payload = json_bytes(cleaned)
    exclusion_payload = json_bytes(exclusions_data)
    buf = io.BytesIO()
    np.save(buf, new_vectors, allow_pickle=False)
    vector_payload = buf.getvalue()
    manifest = {
        "version": "V3-clean-1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": in_manifest.name,
        "source_manifest_sha256": sha256_file(in_manifest),
        "source_chunks_file": in_chunks.name,
        "source_chunks_sha256": sha256_file(in_chunks),
        "source_embeddings_file": in_vectors.name,
        "source_embeddings_sha256": sha256_file(in_vectors),
        "chunks_file": output["chunks"].name,
        "chunks_sha256": sha256_bytes(chunk_payload),
        "embeddings_file": output["vectors"].name,
        "embeddings_sha256": sha256_bytes(vector_payload),
        "exclusions_file": output["exclusions"].name,
        "exclusions_sha256": sha256_bytes(exclusion_payload),
        "model": old_manifest["model"],
        "embedding_dim": 384,
        "source_documents": len({c["document_id"] for c in cleaned}),
        "chunks_before": len(old),
        "chunks_after": len(cleaned),
        "excluded_chunk_ids": [e["original_chunk"]["chunk_id"] for e in exclusions],
        "max_input_tokens_inherited_from_validated_v2": source_count,
        "embeddings_recomputed": False,
        "embedding_alignment": "V3 embedding rows are exact copies of matching V2 row indices.",
        "clinical_content_validation": "NOT_PERFORMED",
        "retrieval_benchmark": "PENDING",
    }

    # Publish manifest LAST: it acts as the completion marker for V3 artifacts.
    store_atomic(output["chunks"], chunk_payload)
    store_atomic(output["vectors"], vector_payload)
    store_atomic(output["exclusions"], exclusion_payload)
    store_atomic(output["manifest"], json_bytes(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing V3")
    args = parser.parse_args()
    result = build(args.data_dir, overwrite=args.overwrite)
    print("STATUS: V3_BUILD_OK (quality review still pending)")
    print(f"Chunks V2 -> V3: {result['chunks_before']} -> {result['chunks_after']}")
    print(f"Documents retained: {result['source_documents']}")
    print(f"Excluded chunk IDs: {result['excluded_chunk_ids']}")
    print("Embedding vectors: reused unchanged, no API or model downloads")
    print("Outputs: medquad_chunks_v3.json, medquad_embeddings_v3.npy,")
    print("         index_exclusions_v3.json, index_manifest_v3.json")


if __name__ == "__main__":
    main()
