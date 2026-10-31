"""Étape 15C: paired regression Vector RAG vs GraphRAG with answer quality controls.

Existing V3 index and Neo4j graph are READ ONLY. Existing step-15 reports untouched.
Only complete outputs with structurally valid references count as successes.

Usage (from 2eme_agent):
  python compare_vector_graphrag_v3_quality.py --dry-run
  python compare_vector_graphrag_v3_quality.py --max-cases 2 --sleep-between 20
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import compare_vector_graphrag_v3 as baseline
import quality_generation_v3 as quality

BASE = Path(__file__).resolve().parent
REPORTS = BASE / "reports" / "hybrid_comparison_v2_quality"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--sleep-between", type=float, default=20)
    args = parser.parse_args()
    if args.max_cases < 1 or args.sleep_between < 0:
        parser.error("--max-cases >= 1 et --sleep-between >= 0 requis")

    import agent_graphrag_hybride_v3 as hybrid
    examples = baseline.load_benchmark()[:args.max_cases]
    baseline.verify_questions(examples, hybrid)
    chunks, vectors = hybrid.load_verified_index(hybrid.DATA)
    by_id = hybrid.check_corpus(chunks)
    print(f"INDEX V3: {len(chunks)} chunks / {len({x['document_id'] for x in chunks})} documents")
    print(f"Benchmark: {len(examples)} questions, mêmes passages vectoriels, modèle {quality.MODEL}")
    if args.dry_run:
        print("DRY-RUN: PASS; pas d'appel Groq, Hugging Face ou Neo4j, pas d'écriture")
        return

    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY absente du fichier .env")

    rows, case_meta = [], []
    for i, ex in enumerate(examples):
        print(f"\nQuestion {i + 1}/{len(examples)}: {ex['id']} | {ex['disease']}", flush=True)
        try:
            context = baseline.collect_case(ex, chunks, vectors, by_id, hybrid)
        except Exception as exc:
            err = f"retrieval:{type(exc).__name__}: {exc}"
            case_meta.append({"case_id": ex["id"], "retrieval_error": err})
            for mode in baseline.MODES:
                rows.append({"case_id": ex["id"], "mode": mode, "question": ex["question"],
                             "disease": ex["disease"], "answer": "", "error": err})
            continue
        case_meta.append({
            "case_id": ex["id"],
            "vector_retrieval_s": round(context["vector_retrieval_s"], 3),
            "graph_retrieval_s": round(context["graph_retrieval_s"], 3),
            "vector_chunk_ids": [x["chunk_id"] for x in context["vector"]],
            "graph_claim_ids": [x["candidate_id"] for x in context["graph"]],
        })
        order = baseline.MODES if i % 2 == 0 else baseline.MODES[::-1]
        for j, mode in enumerate(order):
            evidence, allowed = baseline.build_evidence(mode, context["vector"], context["graph"])
            started = time.perf_counter()
            answer, error, check, attempts = "", None, None, []
            try:
                generation = quality.generate_quality_response(ex["question"], evidence)
                answer, attempts = generation.answer, generation.attempts
                check = baseline.check_citations(answer, allowed, context["graph"], mode)
                if not check["structurally_valid"]:
                    error = "Citations manquantes ou inconnues (contrôle structurel)"
            except quality.IncompleteResponseError as exc:
                attempts = exc.attempts
                error = f"IncompleteResponseError: {exc}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            gen_s = time.perf_counter() - started
            total_s = gen_s + context["vector_retrieval_s"] + (
                context["graph_retrieval_s"] if mode == "hybrid" else 0
            )
            row = {
                "case_id": ex["id"], "mode": mode, "question": ex["question"],
                "disease": ex["disease"], "answer": answer,
                "error": error, "generation_attempts": attempts,
                "completion_verified": error is None,
                "citation_check": check,
                "generation_latency_s": round(gen_s, 3),
                "total_latency_s": round(total_s, 3),
                "vector_chunk_ids": [x["chunk_id"] for x in context["vector"]],
                "graph_claim_ids": ([x["candidate_id"] for x in context["graph"]] if mode == "hybrid" else []),
                "graph_chunk_ids": (sorted({x["evidence_chunk_id"] for x in context["graph"]}) if mode == "hybrid" else []),
            }
            rows.append(row)
            if error:
                print(f"  {mode}: FAILED - {error[:180]}", flush=True)
            else:
                print(f"  {mode}: COMPLETE, citations OK, gen={gen_s:.2f}s, total={total_s:.2f}s", flush=True)
            if args.sleep_between > 0 and (i != len(examples) - 1 or j != len(order) - 1):
                print(f"  Pause réseau : {args.sleep_between:g} s", flush=True)
                time.sleep(args.sleep_between)

    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = REPORTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = baseline.summarize(rows)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "quality_regression_pilot_not_clinically_validated",
        "model": quality.MODEL,
        "generation_budgets": list(quality.DEFAULT_BUDGETS),
        "quality_prompt": quality.SYSTEM,
        "index_chunks_sha256": baseline.sha256_file(hybrid.DATA / "medquad_chunks_v3.json"),
        "index_vectors_sha256": baseline.sha256_file(hybrid.DATA / "medquad_embeddings_v3.npy"),
        "benchmark_sha256": baseline.sha256_file(baseline.BENCHMARK),
        "comparison_contract": "Same question and vector top3; only hybrid adds existing graph facts; same quality prompt/model",
        "limits": ["Same 2 pilot topics", "No independently adjudicated gold answers",
                   "END_RESPONSE and finish_reason do NOT establish semantic completeness",
                   "Citation IDs do NOT validate source support", "Medical correctness unverified"],
        "cases": case_meta, "rows": rows, "summary": summary,
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    baseline.export_review_csv(run_dir / "manual_review.csv", rows)
    md = ["# Étape 15C — Régression qualité Vector vs GraphRAG", "",
          "*Évaluation technique; fidélité médicale et omissions nécessitent une revue humaine indépendante.*", "",
          "| Mode | Réponses terminées + citations structurelles valides | Latence génération | Latence totale |",
          "|---|---:|---:|---:|"]
    for mode, vals in summary.items():
        md.append(f"| {mode} | {vals['successful_generations']}/{vals['attempts']} "
                  f"(citations {vals['valid_citations']}) | {vals['mean_generation_latency_s']} s | "
                  f"{vals['mean_end_to_end_latency_s']} s |")
    md += ["", "Vérifier les réponses dans `manual_review.csv`: nuances d'Addison / secondaire, "
           "distinction most common / other, précision in women, couverture et fidélité de chaque citation.",
           "", "La présence de [[END_RESPONSE]] avant nettoyage et finish_reason=stop "
           "réduit les interruptions non détectées mais ne prouve pas la fidélité."]
    (run_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nRAPPORTS :", run_dir)
    print("  results.json | manual_review.csv | summary.md")
    print("SYNTHÈSE :", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
