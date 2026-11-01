"""Contrôles déterministes du pipeline RAG MedQuAD V1.

À placer dans 2eme_agent/, à côté de rag_pipeline.py.
Ne fait aucun appel à Groq, ne modifie pas les données originales.
Exécution: python audit_pipeline.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPECTED_DIMENSIONS = 384
CHUNK_SIZE_CHARS = 800
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_MAX_TOKENS = 256  # limite utilisée dans la configuration de ce modèle


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        content = json.load(handle)
    if not isinstance(content, list):
        raise ValueError(f"Une liste JSON est attendue dans {path}")
    return content


def main() -> None:
    records = load_json(DATA / "medquad_sample.json")
    chunks = load_json(DATA / "medquad_chunks.json")
    vectors = np.load(DATA / "medquad_embeddings.npy", allow_pickle=False)

    errors: list[str] = []
    warnings: list[str] = []
    ids: list[str] = []
    per_doc: Counter[int] = Counter()

    if len(records) == 0 or len(chunks) == 0:
        errors.append("Les documents ou les chunks sont vides.")

    if vectors.ndim != 2 or vectors.shape != (len(chunks), EXPECTED_DIMENSIONS):
        errors.append(
            f"Dimensions incohérentes : {vectors.shape}, "
            f"attendu ({len(chunks)}, {EXPECTED_DIMENSIONS})."
        )

    if not np.isfinite(vectors).all():
        errors.append("La matrice contient des valeurs NaN ou infinies.")

    # Vérification de la normalisation des embeddings avant la recherche cosinus.
    norm_stats = None
    if vectors.ndim == 2 and vectors.shape[0] and np.isfinite(vectors).all():
        norms = np.linalg.norm(vectors, axis=1)
        norm_stats = {
            "min": float(norms.min()),
            "max": float(norms.max()),
            "mean": float(norms.mean()),
        }
        if not np.allclose(norms, 1.0, rtol=0, atol=1e-3):
            errors.append("Tous les vecteurs ne sont pas normalisés (norme ≈ 1).")
    
    for position, record in enumerate(records):
        for field in ("question", "answer", "source_file", "source_url"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                errors.append(f"Record {position} : champ obligatoire vide/invalide : {field}")

    for position, chunk in enumerate(chunks):
        cid = chunk.get("chunk_id")
        did = chunk.get("document_id")
        content = chunk.get("content")
        ids.append(str(cid))

        if not isinstance(did, int) or not (0 <= did < len(records)):
            errors.append(f"Chunk {position} : document_id invalide ({did}).")
            continue

        per_doc[did] += 1
        original = records[did]

        if not isinstance(content, str) or not content.strip():
            errors.append(f"Chunk {position} : content vide.")
            continue
        if len(content) > CHUNK_SIZE_CHARS:
            errors.append(f"Chunk {position} : dépasse {CHUNK_SIZE_CHARS} caractères.")

        for field in ("question", "source_file", "source_url"):
            if chunk.get(field) != original.get(field):
                errors.append(f"Chunk {position} : métadonnée '{field}' différente du document source.")

        expected_input = f"Question: {original['question']}\nAnswer: {content}"
        if chunk.get("text_for_embedding") != expected_input:
            errors.append(f"Chunk {position} : text_for_embedding ne correspond pas au chunk.")

        if cid != f"{did}_{per_doc[did] - 1}":
            warnings.append(f"Chunk {position} : identifiant ou ordre inattendu ({cid}).")

    if len(ids) != len(set(ids)):
        errors.append("Identifiants chunk_id dupliqués.")

    missing_docs = sorted(set(range(len(records))) - set(per_doc))
    if missing_docs:
        warnings.append(f"Documents sans chunk : {missing_docs}")

    # Détecte des documents similaires textuellement, sans présumer de leur sémantique.
    duplicate_qa_count = len(records) - len({
        (r.get("question", "").strip().lower(), r.get("answer", "").strip().lower())
        for r in records
    })
    if duplicate_qa_count:
        warnings.append(f"{duplicate_qa_count} paires question/réponse identiques.")

    # Contrôle du budget de tokens du modèle. Si le tokenizer n'est pas disponible
    # localement, le script continue et signale l'absence de vérification.
    exceeding_token_budget = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        token_counts = [
            len(tokenizer.encode(c["text_for_embedding"], add_special_tokens=True))
            for c in chunks if isinstance(c.get("text_for_embedding"), str)
        ]
        exceeding_token_budget = sum(n > MODEL_MAX_TOKENS for n in token_counts)
        if exceeding_token_budget:
            warnings.append(
                f"{exceeding_token_budget} chunks dépassent {MODEL_MAX_TOKENS} tokens : "
                "le texte peut être tronqué lors de l'embedding."
            )
    except (ImportError, OSError, ValueError) as exc:
        warnings.append(f"Contrôle des tokens non effectué : {type(exc).__name__}.")

    warnings.append(
        "Vérifier la fraîcheur des sources et les URL historiques MedQuAD."
    )
    warnings.append(
        "Évaluation indépendante nécessaire : les questions originales figurent dans l'index."
    )

    report = {
        "dataset": "MedQuAD/5_NIDDK_QA — extrait de 200 paires",
        "documents": len(records),
        "chunks": len(chunks),
        "embeddings_shape": list(vectors.shape),
        "embedding_model": MODEL_NAME,
        "max_chunk_characters": CHUNK_SIZE_CHARS,
        "embedding_token_limit": MODEL_MAX_TOKENS,
        "chunks_above_token_limit": exceeding_token_budget,
        "vector_norms": norm_stats,
        "documents_without_chunks": missing_docs,
        "errors": errors,
        "warnings": warnings,
        "status": "FAIL" if errors else "PASS_WITH_REVIEW" if warnings else "PASS",
    }

    out = DATA / "pipeline_audit.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Documents : {len(records)} | Chunks : {len(chunks)} | Matrice : {vectors.shape}")
    print(f"Statut : {report['status']}")
    print(f"Erreurs : {len(errors)} | Avertissements : {len(warnings)}")
    for message in errors:
        print("[ERROR]", message)
    for message in warnings[:12]:
        print("[WARN] ", message)
    print("Rapport :", out)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
