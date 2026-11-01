"""Step 10/B: RAGAS v0.4 score generator outputs (Groq via OpenAI-compatible API).

Usage:
    python score_ragas_v3.py --input 'reports/rag_generation_v3/RUN_ID/evaluation_inputs.json' --max-cases 2

Only answerable examples are scored using the four RAGAS metrics.
Unsupported examples are explicitly left for human abstention review.
Requires ragas v0.4, groq, sentence-transformers. Does not update indices.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
EVALUATOR_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def validate_input(data: dict) -> list[dict]:
    examples = data.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("Missing or empty examples")
    seen = set()
    for ex in examples:
        if ex["id"] in seen:
            raise ValueError("Duplicate evaluation ID")
        seen.add(ex["id"])        
        if not ex.get("response", "").strip() or not ex.get("user_input", "").strip():
            raise ValueError(f"Missing response/question for {ex['id']}")
        if not ex.get("retrieved_contexts"):
            raise ValueError(f"Missing contexts for {ex['id']}")
        if ex["type"] == "answerable" and not ex.get("reference", "").strip():
            raise ValueError(f"Missing reference for answerable question {ex['id']}")
    return examples


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to evaluation_inputs.json")
    parser.add_argument("--max-cases", type=int, default=2, help="How many answerable examples to score")
    parser.add_argument("--dry-run", action="store_true", help="Check JSON only, without RAGAS or network")
    args = parser.parse_args()
    if args.max_cases < 1:
        parser.error("--max-cases must be >= 1")
    path = args.input if args.input.is_absolute() else BASE / args.input
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = validate_input(data)
    answerable = [row for row in examples if row["type"] == "answerable"][:args.max_cases]
    unsupported = [row for row in examples if row["type"] != "answerable"]
    print("Input:", path)
    print(f"Answerable selected: {len(answerable)} | unsupported (manual review): {len(unsupported)}")
    if args.dry_run:
        print("DRY RUN: PASS. No model calls.")
        return
    if not answerable:
        raise ValueError("No answerable examples to score")

    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
    from openai import AsyncOpenAI
    from ragas import __version__ as ragas_version
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    )
    from ragas.embeddings import HuggingFaceEmbeddings
    if not ragas_version.startswith("0.4."):
        raise RuntimeError(f"RAGAS v0.4 required by this script; installed: {ragas_version}")
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY missing in project .env")
    # Groq's official OpenAI-compatible endpoint avoids a RAGAS 0.4.3
    # Groq-Instructor adapter bug (client.messages does not exist on Groq()).
    # The API requests still go to Groq and use the existing GROQ_API_KEY.
    groq_compatible_client = AsyncOpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    )
    evaluator_llm = llm_factory(
        EVALUATOR_MODEL,
        provider="openai",
        client=groq_compatible_client,
    )
    evaluator_embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL, device="cpu")
    metrics = {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }
    scored = []
    for ex in answerable:
        print("Scoring:", ex["id"])
        inputs = {"user_input": ex["user_input"], "response": ex["response"],
                  "retrieved_contexts": ex["retrieved_contexts"], "reference": ex["reference"]}
        values = {}
        errors = {}
        for key, metric in metrics.items():
            try:
                if key == "faithfulness":
                    kwargs = {k: inputs[k] for k in ("user_input", "response", "retrieved_contexts")}
                elif key == "answer_relevancy":
                    kwargs = {k: inputs[k] for k in ("user_input", "response")}
                elif key == "context_precision":
                    kwargs = {k: inputs[k] for k in ("user_input", "reference", "retrieved_contexts")}
                else:
                    kwargs = {k: inputs[k] for k in ("user_input", "reference", "retrieved_contexts")}
                result = metric.score(**kwargs)
                val = float(result.value)
                if not math.isfinite(val):
                    raise ValueError("Non-finite RAGAS score")
                values[key] = val
                print(f"  {key}: {val:.4f}")
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {exc}"
                values[key] = None
                print(f"  {key}: ERROR - {type(exc).__name__}")
        scored.append({"id": ex["id"], "scores": values, "errors": errors})
    averages = {
        name: mean_or_none([row["scores"][name] for row in scored
                            if row["scores"][name] is not None]) for name in metrics
    }
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(path), "ragas_version": ragas_version,
        "evaluator_model": EVALUATOR_MODEL, "embedding_model": EMBEDDING_MODEL,
        "status": "exploratory_scores_not_clinically_validated",
        "scored": scored, "mean_scores": averages,
        "unsupported_cases_for_manual_review": [
            {"id": ex["id"], "user_input": ex["user_input"], "response": ex["response"]}
            for ex in unsupported
        ],
        "note": "Missing scores (null) indicate judge errors, not zero or passing scores. No medical validity inference.",
    }
    out = path.parent / "ragas_scores.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out)
    print("Mean scores (available only):", averages)
    print("Inspect errors and unsupported responses manually before drawing conclusions.")


if __name__ == "__main__":
    main()