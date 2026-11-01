
import json
import random
from pathlib import Path

from rag_pipeline import retrieve, answer_question

# ======================================
# 1. LOAD EVALUATION DATA
# ======================================

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

with open(
    DATA / "medquad_sample.json",
    "r",
    encoding="utf-8"
) as f:
    records = json.load(f)

# Choose 20 reproducible test questions
random_generator = random.Random(42)

test_ids = random_generator.sample(
    range(len(records)),
    k=min(20, len(records))
)

# ======================================
# 2. EVALUATE RETRIEVAL
# ======================================

hits_at_1 = 0
hits_at_3 = 0
reciprocal_rank_sum = 0

print("\n=== RETRIEVAL EVALUATION ===")

for doc_id in test_ids:

    question = records[doc_id]["question"]

    results = retrieve(question, top_k=3)

    retrieved_ids = [
        doc["document_id"]
        for doc in results
    ]

    # Find the rank of the expected document
    rank = None

    for position, retrieved_id in enumerate(
        retrieved_ids, start=1
    ):
        if retrieved_id == doc_id:
            rank = position
            break

    if rank == 1:
        hits_at_1 += 1

    if rank is not None:
        hits_at_3 += 1
        reciprocal_rank_sum += 1 / rank

    print(
        f"Document {doc_id} | "
        f"Expected rank: {rank} | "
        f"Retrieved: {retrieved_ids}"
    )

n = len(test_ids)

print("\n=== METRICS ===")
print(f"Hit@1 : {hits_at_1 / n:.2%}")
print(f"Hit@3 : {hits_at_3 / n:.2%}")
print(f"MRR@3 : {reciprocal_rank_sum / n:.4f}")

# ======================================
# 3. GENERATION EVALUATION
# ======================================

questions = [
    "What are the symptoms of acromegaly?",
    "What causes acromegaly?",
    "What do the 2026 clinical guidelines "
    "recommend for acromegaly?"
]

report_path = DATA / "rag_evaluation_report.md"

with open(
    report_path, "w", encoding="utf-8"
) as report:

    report.write("# RAG Evaluation Report\n\n")

    for i, question in enumerate(
        questions, start=1
    ):

        print(f"\nGenerating answer {i}/3")

        answer, sources = answer_question(question)

        report.write(f"## Test {i}\n\n")
        report.write(f"**Question:** {question}\n\n")
        report.write(f"**LLM Answer:**\n\n{answer}\n\n")
        report.write("### Retrieved Context\n\n")

        for j, source in enumerate(
            sources, start=1
        ):
            report.write(
                f"#### Source [{j}]\n\n"
                f"Score: {source['score']:.4f}\n\n"
                f"Original question: "
                f"{source['question']}\n\n"
                f"{source['content']}\n\n"
                f"URL: {source['source_url']}\n\n"
            )

        report.write(
            "**Manual Faithfulness:** À vérifier\n\n"
        )
        report.write(
            "**Manual Answer Relevance:** À vérifier\n\n"
        )
        report.write("---\n\n")

print("\nReport generated:", report_path)
