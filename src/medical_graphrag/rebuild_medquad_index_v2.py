r"""Correct the token-budget issue in MedQuAD index V1 without altering baseline files.

Place in 2eme_agent/ and run using that project's .venv:
    .\.venv\Scripts\python.exe .\rebuild_medquad_index_v2.py

Input:  data/medquad_chunks.json
Output: data/medquad_chunks_v2.json
        data/medquad_embeddings_v2.npy
        data/index_manifest_v2.json

This is a supervised indexing operation, NOT a medical evaluation.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OLD_CHUNKS = DATA / "medquad_chunks.json"
NEW_CHUNKS = DATA / "medquad_chunks_v2.json"
NEW_VECTORS = DATA / "medquad_embeddings_v2.npy"
MANIFEST = DATA / "index_manifest_v2.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_MODEL_TOKENS = 256
MAX_CHUNK_CHARS = 800


def token_length(tokenizer, text: str) -> int:
    # truncation=False is essential: we are detecting potential truncation.
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def split_oversized_chunk(tokenizer, question: str, content: str) -> list[str]:
    """Retain natural separators, and validate the *entire* embedding input."""
    prefix = f"Question: {question}\nAnswer: "
    budget = min(210, MAX_MODEL_TOKENS - token_length(tokenizer, prefix) - 12)
    if budget < 32:
        raise ValueError("Question too long to preserve a useful answer token budget.")

    while budget >= 32:
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=budget,
            chunk_overlap=min(24, budget // 8),
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        parts = splitter.split_text(content)
        if parts and all(
            p.strip()
            and len(p) <= MAX_CHUNK_CHARS
            and token_length(tokenizer, prefix + p) <= MAX_MODEL_TOKENS
            for p in parts
        ):
            return parts
        budget -= 8

    raise RuntimeError("Could not split text within token and character limits.")


def main() -> None:
    old = json.loads(OLD_CHUNKS.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    new_chunks: list[dict] = []
    per_doc: defaultdict[int, int] = defaultdict(int)
    oversized: list[tuple[str, int]] = []

    for original in old:
        question = original["question"]
        content = original["content"]
        embedding_text = original["text_for_embedding"]
        expected_text = f"Question: {question}\nAnswer: {content}"
        if embedding_text != expected_text:
            raise ValueError(f"Invalid input payload in {original['chunk_id']}")
        if len(content) > MAX_CHUNK_CHARS:
            raise ValueError(f"Existing content exceeds 800 chars: {original['chunk_id']}")

        n_tokens = token_length(tokenizer, embedding_text)
        if n_tokens > MAX_MODEL_TOKENS:
            oversized.append((original["chunk_id"], n_tokens))
            parts = split_oversized_chunk(tokenizer, question, content)
        else:
            parts = [content]

        for part in parts:
            row = dict(original)
            doc_id = int(row["document_id"])
            row["chunk_id"] = f"{doc_id}_{per_doc[doc_id]}"
            row["content"] = part
            row["text_for_embedding"] = f"Question: {question}\nAnswer: {part}"
            per_doc[doc_id] += 1
            new_chunks.append(row)

    final_lengths = [token_length(tokenizer, c["text_for_embedding"]) for c in new_chunks]
    if not new_chunks or max(final_lengths) > MAX_MODEL_TOKENS:
        raise AssertionError("Some V2 chunks still exceed the model token limit.")
    if len({c["chunk_id"] for c in new_chunks}) != len(new_chunks):
        raise AssertionError("Duplicate V2 chunk IDs.")
    if len(per_doc) != len({c["document_id"] for c in old}):
        raise AssertionError("Documents were lost during re-chunking.")

    print("V1 chunks:", len(old))
    print("Excess-token V1 chunks:", oversized)
    print("V2 chunks:", len(new_chunks))
    print("V2 max tokens:", max(final_lengths))
    print("Generating V2 embeddings with same model/configuration...")

    embedder = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    vectors = np.asarray(
        embedder.embed_documents([c["text_for_embedding"] for c in new_chunks]),
        dtype=np.float32,
    )
    if vectors.shape != (len(new_chunks), 384):
        raise AssertionError(f"Invalid embeddings shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise AssertionError("Non-finite embeddings")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=0, atol=1e-3):
        raise AssertionError("Embeddings are not normalized")

    # New versioned output names preserve the V1 baseline for comparisons.
    new_chunk_bytes = json.dumps(new_chunks, indent=2, ensure_ascii=False).encode("utf-8")
    NEW_CHUNKS.write_bytes(new_chunk_bytes)
    with NEW_VECTORS.open("wb") as f:
        np.save(f, vectors, allow_pickle=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MedQuAD NIDDK sample",
        "source_chunks_file": OLD_CHUNKS.name,
        "source_chunks_sha256": hashlib.sha256(OLD_CHUNKS.read_bytes()).hexdigest(),
        "chunks_file": NEW_CHUNKS.name,
        "chunks_sha256": hashlib.sha256(new_chunk_bytes).hexdigest(),
        "embeddings_file": NEW_VECTORS.name,
        "model": MODEL_NAME,
        "max_model_tokens_including_special_tokens": MAX_MODEL_TOKENS,
        "max_input_tokens_observed": max(final_lengths),
        "chunks_before": len(old),
        "chunks_after": len(new_chunks),
        "oversized_v1_chunks": oversized,
        "remaining_oversized_v2_chunks": 0,
        "embedding_dimensions": 384,
        "normalization": "L2, unit length",
        "note": "This is a versioned technical fix, not a retrieval-quality improvement claim.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", NEW_CHUNKS)
    print("Saved:", NEW_VECTORS)
    
    print("Saved:", MANIFEST)


if __name__ == "__main__":
    main()
