"""Pilote GraphRAG V3 avec contrôles de complétude et nuances documentaires.

Usage:
  python agent_graphrag_hybride_v3_quality.py --check
  python agent_graphrag_hybride_v3_quality.py --question "Quels sont les symptômes de l'acromégalie ?"

Conserve l'index V3 et Neo4j intacts. Refuse les réponses incomplètes.
"""
from __future__ import annotations
import argparse
import os

from medical_graphrag.agents import agent_graphrag_hybride_v3 as hybrid
from medical_graphrag.evaluation import quality_generation_v3 as quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--question", type=str)
    args = parser.parse_args()
    chunks, vectors = hybrid.load_verified_index(hybrid.DATA)
    by_id = hybrid.check_corpus(chunks)
    print(f"INDEX V3 OK : {len(chunks)} chunks, {len({x['document_id'] for x in chunks})} documents")
    if args.check:
        print("PASS : contrôle local sans appel à Groq, HF ou Neo4j")
        return
    question = args.question
    guard = hybrid.preflight_block(question)
    if guard:
        print("RÉPONSE DOCUMENTAIRE LIMITÉE :", guard)
        return
    disease = hybrid.canonical_disease(question)
    if not disease or not hybrid.symptoms_intent(question):
        print("Pilote limité aux symptômes de l'acromégalie et de l'insuffisance surrénalienne.")
        return
    from dotenv import load_dotenv
    load_dotenv(hybrid.BASE / ".env")
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY absente du fichier .env")

    history = []
    def synthesize(q: str, graph: list[dict], vector: list[dict]) -> str:
        evidence, _ = __import__("medical_graphrag.evaluation.compare_vector_graphrag_v3", fromlist=["*"]).build_evidence("hybrid", vector, graph)
        output = quality.generate_quality_response(q, evidence)
        history.extend(output.attempts)
        return output.answer

    workflow = hybrid.build_workflow(
        lambda name: hybrid.lookup_graph_live(name, by_id),
        lambda q: hybrid.semantic_search(q, chunks, vectors, top_k=3),
        synthesize,
    )
    state = workflow.invoke({"question": question, "disease": disease})
    print("\nOUTILS : Neo4j + Vector V3 + synthèse Groq")
    print("PREUVES NEO4J :", ", ".join(x["candidate_id"] for x in state["graph_evidence"]))
    print("PASSAGES VECTORIELS :", ", ".join(x["chunk_id"] for x in state["vector_evidence"]))
    print("FIN DE RÉPONSE CONTRÔLÉE :", history[-1])
    print("\nRÉPONSE :\n", state["answer"])
    print("\nCITATIONS :", ", ".join("[" + c + "]" for c in state["cited_chunks"]))
    print("AVERTISSEMENT : citations et complétude structurelles seulement; aucune validation clinique.")


if __name__ == "__main__":
    main()
