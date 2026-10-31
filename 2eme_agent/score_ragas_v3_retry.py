"""RAGAS 0.4.x: targeted, rate-limit-aware evaluation of saved MedQuAD V3 responses.

Run from 2eme_agent:
    python score_ragas_v3_retry.py --input "reports/rag_generation_v3/RUN_ID/evaluation_inputs.json" --dry-run
    python score_ragas_v3_retry.py --input "reports/rag_generation_v3/RUN_ID/evaluation_inputs.json" --metrics faithfulness --max-cases 1
    python score_ragas_v3_retry.py --input "reports/rag_generation_v3/RUN_ID/evaluation_inputs.json" --metrics all --max-cases 2

Uses Groq via its OpenAI-compatible endpoint, with a real async client.
No modification to the corpus, embeddings or saved generated responses.
RAGAS metric values are not medical-validity scores.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
EVALUATOR_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def validate_input(data: dict) -> list[dict]:
    examples = data.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("Missing or empty 'examples' array")
    seen = set()
    for ex in examples:
        if not isinstance(ex, dict):
            raise ValueError("An example is not a dictionary")
        case_id = ex.get("id")
        if not case_id or case_id in seen:
            raise ValueError("Missing or duplicate evaluation ID")
        seen.add(case_id)
        if not isinstance(ex.get("user_input"), str) or not ex["user_input"].strip():
            raise ValueError(f"Missing question: {case_id}")
        if not isinstance(ex.get("response"), str) or not ex["response"].strip():
            raise ValueError(f"Missing generated response: {case_id}")
        contexts = ex.get("retrieved_contexts")
        if not isinstance(contexts, list) or not contexts or any(not isinstance(x, str) for x in contexts):
            raise ValueError(f"Invalid retrieved contexts: {case_id}")
        if ex.get("type") == "answerable" and not str(ex.get("reference", "")).strip():
            raise ValueError(f"Missing reference: {case_id}")
    return examples


def is_rate_limited(exc: Exception) -> bool:
    """Instructor wraps OpenAI exceptions: check the chain and message."""
    cur = exc
    seen = set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        status = getattr(cur, "status_code", None)
        message = str(cur).lower()
        if status == 429 or "error code: 429" in message or "rate_limit_exceeded" in message:
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--max-cases", default=1, type=int)
    parser.add_argument("--metrics", choices=("all",) + METRICS, default="faithfulness")
    parser.add_argument("--max-output-tokens", default=2200, type=int,
                        help="Max generated tokens per judge request; larger consumes more TPM")
    parser.add_argument("--sleep-between", default=20.0, type=float,
                        help="Minimum pause (seconds) between different metric evaluations")
    parser.add_argument("--rate-retries", default=2, type=int,
                        help="Max retries per metric after Groq HTTP 429")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_cases < 1 or args.max_output_tokens < 256 or args.sleep_between < 0 or not (0 <= args.rate_retries <= 5):
        parser.error("Invalid max-cases / max-output-tokens / sleep-between / rate-retries")
    selected = METRICS if args.metrics == "all" else (args.metrics,)
    path = args.input if args.input.is_absolute() else BASE / args.input
    if not path.is_file():
        raise FileNotFoundError(path)
    examples = validate_input(json.loads(path.read_text(encoding="utf-8")))
    answerable = [ex for ex in examples if ex.get("type") == "answerable"][:args.max_cases]
    unsupported = [ex for ex in examples if ex.get("type") != "answerable"]
    print(f"Input: {path}")
    print(f"Selected: {len(answerable)} answerable, {len(unsupported)} unsupported for manual review")
    print(f"Metrics: {', '.join(selected)} | output budget: {args.max_output_tokens} tokens")
    if args.dry_run:
        print("DRY RUN: PASS. No API or embedding model calls.")
        return
    if not answerable:
        raise ValueError("No answerable examples selected")

    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY missing in .env")
    from openai import AsyncOpenAI
    from ragas import __version__ as ragas_version
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    if not ragas_version.startswith("0.4."):
        raise RuntimeError(f"Expected ragas 0.4.x, found {ragas_version}")

    sdk_client = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        timeout=90.0,
        max_retries=0,  # outer loop handles 429; avoid hidden duplicate calls
    )
    llm = llm_factory(
        EVALUATOR_MODEL,
        provider="openai",
        client=sdk_client,
        temperature=0.01,
        top_p=0.1,
        max_tokens=args.max_output_tokens,
        extra_body={"reasoning_effort": "low", "reasoning_format": "hidden"},
    )
    if getattr(llm, "is_async", False) is not True:
        raise RuntimeError("RAGAS evaluator is not asynchronous")
    embedding_model = None
    if "answer_relevancy" in selected:
        from ragas.embeddings import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(model=EMBEDDING_MODEL, device="cpu")
    metrics = {}
    for name in selected:
        if name == "faithfulness":
            metrics[name] = Faithfulness(llm=llm)
        elif name == "answer_relevancy":
            metrics[name] = AnswerRelevancy(llm=llm, embeddings=embedding_model)
        elif name == "context_precision":
            metrics[name] = ContextPrecision(llm=llm)
        else:
            metrics[name] = ContextRecall(llm=llm)

    scored = []
    first_call = True
    for ex in answerable:
        print("Scoring:", ex["id"])
        values, errors = {}, {}
        for name, metric in metrics.items():
            if not first_call and args.sleep_between:
                time.sleep(args.sleep_between)
            first_call = False
            if name == "faithfulness":
                kwargs = {k: ex[k] for k in ("user_input", "response", "retrieved_contexts")}
            elif name == "answer_relevancy":
                kwargs = {k: ex[k] for k in ("user_input", "response")}
            else:
                kwargs = {k: ex[k] for k in ("user_input", "reference", "retrieved_contexts")}
            for attempt in range(args.rate_retries + 1):
                try:
                    result = metric.score(**kwargs)
                    value = float(result.value)
                    if not math.isfinite(value):
                        raise ValueError("Non-finite score")
                    values[name] = value
                    print(f"  {name}: {value:.4f}")
                    break
                except Exception as exc:
                    if is_rate_limited(exc) and attempt < args.rate_retries:
                        delay = 30 * (attempt + 1)
                        print(f"  {name}: HTTP 429; retry {attempt + 1}/{args.rate_retries} after {delay}s")
                        time.sleep(delay)
                        continue
                    values[name] = None
                    errors[name] = f"{type(exc).__name__}: {str(exc)[:1200]}"
                    print(f"  {name}: ERROR - {errors[name][:350]}")
                    break
        scored.append({"id": ex["id"], "scores": values, "errors": errors})

    means = {name: (
        sum(row["scores"][name] for row in scored if row["scores"][name] is not None)
        / sum(row["scores"][name] is not None for row in scored)
        if any(row["scores"][name] is not None for row in scored) else None
    ) for name in selected}
    successful = all(v is not None for row in scored for v in row["scores"].values())
    status = "selected_metrics_succeeded" if successful else "partial_or_failed_evaluation"
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input": str(path),
        "ragas_version": ragas_version,
        "evaluator_model": EVALUATOR_MODEL,
        "model_api_provider": "groq_openai_compatible",
        "embedding_model": EMBEDDING_MODEL if embedding_model is not None else None,
        "metrics_requested": list(selected),
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": "low",
        "sleep_between": args.sleep_between,
        "scored": scored,
        "mean_scores": means,
        "unsupported_for_manual_review": [
            {"id": ex["id"], "user_input": ex["user_input"], "response": ex["response"]}
            for ex in unsupported
        ],
        "note": "Only requested metrics are scored; null means error. Scores do not certify medical accuracy.",
    }
    output = path.parent / ("ragas_retry_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved:", output)
    print("Mean scores:", means)
    print("Status:", status)


if __name__ == "__main__":
    main()
