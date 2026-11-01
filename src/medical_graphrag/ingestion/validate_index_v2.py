"""Validation QA de l'index MedQuAD V2, sans modifier les index existants.

Dans le dossier 2eme_agent :
    .\\.venv\\Scripts\\python.exe .\\validate_index_v2.py
Sortie : data/index_v2_quality_report.json
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
V1_FILE = DATA / "medquad_chunks.json"
CHUNKS_FILE = DATA / "medquad_chunks_v2.json"
VECTORS_FILE = DATA / "medquad_embeddings_v2.npy"
MANIFEST_FILE = DATA / "index_manifest_v2.json"
REPORT_FILE = DATA / "index_v2_quality_report.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN_LIMIT = 256
CHAR_LIMIT = 800


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    errors, warnings = [], []
    required = [V1_FILE, CHUNKS_FILE, VECTORS_FILE, MANIFEST_FILE]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Fichiers manquants : " + ", ".join(missing))

    v1 = json.loads(V1_FILE.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    vectors = np.load(VECTORS_FILE, allow_pickle=False)

    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        errors.append("Identifiants chunk_id dupliqués.")
    if len(chunks) != vectors.shape[0] or vectors.ndim != 2 or vectors.shape[1] != 384:
        errors.append(f"Matrice incompatible : {vectors.shape}, chunks={len(chunks)}.")
    if not np.isfinite(vectors).all():
        errors.append("Valeurs non finies dans les embeddings.")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3, rtol=0):
        errors.append("Certains vecteurs ne sont pas normalisés.")

    if manifest.get("chunks_sha256") != sha256(CHUNKS_FILE):
        errors.append("Empreinte SHA-256 du JSON V2 incohérente avec le manifeste.")
    if manifest.get("source_chunks_sha256") != sha256(V1_FILE):
        errors.append("Le fichier d'entrée V1 diffère de la version référencée.")
    if manifest.get("chunks_after") != len(chunks):
        errors.append("Nombre de chunks V2 incohérent dans le manifeste.")

    source_meta = {
        (c["document_id"], c["question"], c["source_file"], c["source_url"])
        for c in v1
    }
    actual_docs = {c["document_id"] for c in chunks}
    source_docs = {c["document_id"] for c in v1}
    if actual_docs != source_docs:
        errors.append("Documents perdus ou supplémentaires dans V2.")

    malformed, long_chars = [], []
    for c in chunks:
        if (c["document_id"], c["question"], c["source_file"], c["source_url"]) not in source_meta:
            malformed.append(c["chunk_id"])
        if c["text_for_embedding"] != f"Question: {c['question']}\nAnswer: {c['content']}":
            malformed.append(c["chunk_id"])
        if not c["content"].strip() or len(c["content"]) > CHAR_LIMIT:
            long_chars.append(c["chunk_id"])
    if malformed:
        errors.append(f"Métadonnées / payload incorrects : {sorted(set(malformed))[:10]}")
    if long_chars:
        errors.append(f"Chunks vides ou trop longs en caractères : {long_chars[:10]}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    lengths = [len(tokenizer.encode(c["text_for_embedding"], add_special_tokens=True, truncation=False)) for c in chunks]
    oversize = [chunks[i]["chunk_id"] for i, length in enumerate(lengths) if length > TOKEN_LIMIT]
    if oversize:
        errors.append(f"Chunks >{TOKEN_LIMIT} tokens : {oversize[:10]}")

    # Ces fragments ne bloquent pas l'index technique, mais nuisent à la pertinence.
    short_fragments = [
        {"chunk_id": c["chunk_id"], "text": c["content"], "n_characters": len(c["content"])}
        for c in chunks if len(c["content"].strip()) <= 20
    ]
    if short_fragments:
        warnings.append(f"{len(short_fragments)} chunks très courts à examiner manuellement.")
    warnings.append("La source MedQuAD est historique : sa fraîcheur clinique n'est pas validée.")
    warnings.append("Le benchmark indépendant reste à construire.")

    report = {
        "index_version": "V2",
        "status": "FAIL" if errors else "PASS_WITH_REVIEW" if warnings else "PASS",
        "source_documents": len(source_docs),
        "v1_chunks": len(v1),
        "v2_chunks": len(chunks),
        "embeddings_shape": list(vectors.shape),
        "max_embedding_input_tokens": max(lengths) if lengths else None,
        "chunks_above_token_limit": len(oversize),
        "very_short_fragments": short_fragments,
        "errors": errors,
        "warnings": warnings,
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Statut :", report["status"])
    print("Documents :", report["source_documents"])
    print("Chunks V1 / V2 :", len(v1), "/", len(chunks))
    print("Matrice :", vectors.shape)
    print("Tokens max :", report["max_embedding_input_tokens"])
    print("Erreurs :", errors)
    print("Fragments courts :", short_fragments)
    print("Rapport :", REPORT_FILE)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()