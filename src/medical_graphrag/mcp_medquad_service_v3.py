"""MedQuAD V3 MCP tool implementations, kept independent of the MCP SDK.

Reads only. Historical educational corpus; not medical advice. No arbitrary Cypher.
The three functions can be called by a protocol server or by unit tests.
"""
from __future__ import annotations

import sys
from contextlib import redirect_stdout
from functools import lru_cache
from typing import Any

MAX_QUESTION_CHARS = 500
MAX_TOP_K = 5


def _hybrid():
    # Lazy imports: listing MCP tools must not load models or contact Neo4j.
    import agent_graphrag_hybride_v3 as hybrid
    return hybrid


@lru_cache(maxsize=1)
def _verified_index():
    hybrid = _hybrid()
    chunks, vectors = hybrid.load_verified_index(hybrid.DATA)
    hybrid.check_corpus(chunks)
    return chunks, vectors


def corpus_stats() -> dict[str, Any]:
    """Stats from the locally verified MedQuAD V3 files (no API calls)."""
    chunks, vectors = _verified_index()
    return {
        "status": "ok", "index_version": "V3",
        "documents": len({r["document_id"] for r in chunks}),
        "chunks": len(chunks),
        "vector_dimensions": int(vectors.shape[1]),
        "scope": "historical_medquad_documents_not_clinical_guidelines",
    }


def vector_search(question: str, top_k: int = 3) -> dict[str, Any]:
    """Semantic search over V3. Question is data, never code or instructions."""
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION_CHARS:
        raise ValueError("Question vide ou >500 caractères")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not (1 <= top_k <= MAX_TOP_K):
        raise ValueError("top_k doit être un entier de 1 à 5")
    hybrid = _hybrid()
    if (blocked := hybrid.preflight_block(question)):
        return {"status": "not_supported", "reason": blocked, "results": []}
    chunks, vectors = _verified_index()
    # Protect MCP stdio: some model dependencies write progress to stdout.
    with redirect_stdout(sys.stderr):
        matches = hybrid.semantic_search(question, chunks, vectors, top_k=top_k)
    return {
        "status": "ok", "index_version": "V3", "question": question,
        "top_k": top_k, "results": matches,
        "warning": "Similarité non équivalente à une probabilité ou une validation médicale.",
    }


def graph_symptoms(disease: str) -> dict[str, Any]:
    """Fixed, parameterized read-only Cypher query with chunk provenance checks."""
    if not isinstance(disease, str) or not disease.strip() or len(disease) > 120:
        raise ValueError("Nom de maladie vide ou trop long")
    hybrid = _hybrid()
    canonical = hybrid.canonical_disease(disease)
    if not canonical:
        return {
            "status": "not_supported", "disease_requested": disease,
            "claims": [], "reason": "Pilote limité à Acromegaly et Adrenal insufficiency",
        }
    from dotenv import load_dotenv
    load_dotenv(hybrid.BASE / ".env")
    chunks, _ = _verified_index()
    by_id = hybrid.check_corpus(chunks)
    # lookup_graph_live uses a fixed query, verifies 8/6 claims, and checks all
    # source URLs, document/chunk links and evidence hashes. No Groq call.
    claims = hybrid.lookup_graph_live(canonical, by_id)
    return {
        "status": "ok", "index_version": "V3", "disease": canonical,
        "claims_count": len(claims), "claims": claims,
        "warning": "Claims documentaires historiques, pas des connaissances cliniques validées.",
    }
