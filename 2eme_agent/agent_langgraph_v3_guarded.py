"""Étape 12 — Contrôles déterministes autour de l'agent LangGraph V3.

Dépend du fichier déjà présent : agent_langgraph_v3.py.
N'altère ni les index, ni le fichier .env, ni les autres scripts.

Exemples (depuis 2eme_agent) :
  python agent_langgraph_v3_guarded.py --check
  python agent_langgraph_v3_guarded.py --question "What are the symptoms of acromegaly?"
  python agent_langgraph_v3_guarded.py --question "What do the latest 2026 clinical guidelines recommend for acromegaly?"

Ce programme ne valide PAS la vérité médicale des affirmations : il vérifie seulement
la correspondance question-outil et l'existence des références citées.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from typing import Any

CITE_RE = re.compile(r"\[(\d+_\d+)\]|【(\d+_\d+)】")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def classify_question(question: str) -> str:
    """Un routeur limité et vérifiable, adapté aux cas de test du prototype.

    Ne remplace PAS une classification clinique ou un filtre de sécurité complet.
    """
    q = normalize(question).replace("’", "'")
    if not q.strip():
        return "empty"
    clinical = re.search(
        r"\b(guidelines?|recommandations?|clinical|clinique|cliniques|"
        r"treatments?|traitements?|protocols?|protocoles?|"
        r"therapies|therapy|therapeutiques?)\b", q
    )
    recent = re.search(
        r"\b(latest|current|recent|updated|newest|today|now|actuel|actuelle|"
        r"actuelles|actuels|recentes?|dernieres?|nouveaux|nouvelles|"
        r"202[4-9]|20[3-9]\d|a jour)\b", q
    )
    if clinical and recent:
        return "current_clinical"
    if re.search(
        r"\b(should i take|should i use|what dose should i|my symptoms|"
        r"i have been diagnosed|i have symptoms|diagnose me|prescribe me|"
        r"quel medicament dois je|quelle dose dois je|mes symptomes|"
        r"j'ai des symptomes|j'ai ete diagnostique|mon traitement)\b", q
    ):
        return "personal_medical"
    statistics_subject = re.search(r"\b(documents?|chunks?|corpus|index|indexed|dataset|embeddings?)\b", q)
    statistics_request = re.search(
        r"\b(how many|number|count|statistics|stats|statistiques|combien|"
        r"nombre|dimension|dimensions|which model|what model|quel modele)\b", q
    )
    if statistics_subject and statistics_request:
        return "statistics"
    return "search"


def preflight_block(question: str) -> str | None:
    kind = classify_question(question)
    french = bool(re.search(r"\b(que|quelles?|combien|pourquoi|je|mon|mes|recommandations|actuelles)\b", normalize(question)))
    if kind == "empty":
        return "Question vide." if french else "Empty question."
    if kind == "current_clinical":
        if french:
            return ("Ce prototype utilise des textes MedQuAD historiques. Il ne peut pas "
                    "vérifier les recommandations cliniques actuelles. Consultez une source "
                    "médicale officielle récente ou un professionnel de santé.")
        return ("This prototype uses historical MedQuAD documents and cannot verify "
                "current clinical guidelines. Consult an up-to-date official medical "
                "source or a qualified healthcare professional.")
    if kind == "personal_medical":
        if french:
            return ("Ce prototype de recherche documentaire ne fournit ni diagnostic "
                    "personnalisé, ni prescription. Consultez un professionnel de santé.")
        return ("This document-research prototype cannot provide a personal diagnosis "
                "or prescription. Consult a qualified healthcare professional.")
    return None


def validate_execution(question: str, result: dict[str, Any]) -> dict[str, Any]:
    """Valide la trace de l'agent, indépendamment de la justesse médicale.

    Accepte des messages LangChain (type='tool'/'ai') ou des objets équivalents.
    L'enforcement d'une politique complète ne peut pas se réduire à ces regex.
    """
    if preflight_block(question):
        return {"allowed": False, "reason": "preflight_should_have_blocked", "answer": "", "sources": []}

    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        return {"allowed": False, "reason": "missing_messages", "answer": "", "sources": []}

    tools = [msg for msg in messages if getattr(msg, "type", None) == "tool"]
    tool_names = {getattr(msg, "name", "") for msg in tools}
    kind = classify_question(question)
    required_tool = "statistiques_corpus" if kind == "statistics" else "rechercher_medquad"
    if required_tool not in tool_names:
        return {"allowed": False, "reason": f"required_tool_missing:{required_tool}", "answer": "", "sources": []}

    final_msg = messages[-1]
    if getattr(final_msg, "type", None) != "ai" or getattr(final_msg, "tool_calls", None):
        return {"allowed": False, "reason": "no_final_ai_message", "answer": "", "sources": []}
    answer = getattr(final_msg, "content", "")
    if not isinstance(answer, str) or not answer.strip():
        return {"allowed": False, "reason": "empty_or_nontext_answer", "answer": "", "sources": []}

    if kind == "statistics":
        return {"allowed": True, "reason": "statistics_tool_used", "answer": answer, "sources": []}

    evidence: dict[str, str] = {}
    for msg in tools:
        if getattr(msg, "name", "") != "rechercher_medquad":
            continue
        try:
            payload = json.loads(msg.content)
            for hit in payload.get("results", []):
                cid = hit["chunk_id"]
                if not isinstance(cid, str) or not re.fullmatch(r"\d+_\d+", cid):
                    raise ValueError("invalid chunk id")
                evidence[cid] = str(hit.get("source_url_historical", ""))
        except (ValueError, KeyError, TypeError, AttributeError):
            return {"allowed": False, "reason": "invalid_retrieval_trace", "answer": "", "sources": []}
    if not evidence:
        return {"allowed": False, "reason": "no_retrieved_evidence", "answer": "", "sources": []}
    cited = {left or right for left, right in CITE_RE.findall(answer)}
    if not cited:
        return {"allowed": False, "reason": "no_chunk_citation_in_answer", "answer": "", "sources": []}
    unknown = sorted(cited - evidence.keys())
    if unknown:
        return {"allowed": False, "reason": "unknown_citations:" + ",".join(unknown), "answer": "", "sources": []}
    return {
        "allowed": True,
        "reason": "search_used_and_citations_traceable",
        "answer": answer,
        "sources": [(cid, evidence[cid]) for cid in sorted(cited)],
    }


def run_guarded(graph: Any, question: str) -> None:
    reason = preflight_block(question)
    if reason:
        print("\nRÉPONSE DOCUMENTAIRE LIMITÉE :", reason)
        print("OUTILS UTILISÉS : aucun (blocage déterministe avant appel API)")
        return
    result = graph.invoke({"messages": [("user", question)]}, config={"recursion_limit": 10})
    decision = validate_execution(question, result)
    used = [getattr(m, "name", "") for m in result.get("messages", []) if getattr(m, "type", None) == "tool"]
    print("\nOUTILS UTILISÉS :", ", ".join(used) if used else "aucun")
    if not decision["allowed"]:
        print("\nRÉPONSE BLOQUÉE PAR VALIDATION :", decision["reason"])
        return
    print("\nRÉPONSE :\n", decision["answer"])
    if decision["sources"]:
        print("\nRÉFÉRENCES CITÉES ET RETROUVÉES (existence vérifiée, contenu NON validé) :")
        for chunk_id, historical_url in decision["sources"]:
            print(f"  [{chunk_id}] {historical_url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--question", type=str)
    args = parser.parse_args()
    if not args.check and args.question is None:
        parser.error("Utiliser --check ou --question '...'")
    from agent_langgraph_v3 import load_verified_index, build_agent, DATA
    chunks, vectors = load_verified_index(DATA)
    print(f"INDEX V3 OK : {len(chunks)} chunks, {len({x['document_id'] for x in chunks})} documents, vecteurs {vectors.shape}")
    if args.check:
        print("PASS : intégrité V3. Aucun appel LLM.")
        return
    # Traiter les demandes non couvertes sans appeler l'API Groq.
    rejection = preflight_block(args.question)
    if rejection:
        print("\nRÉPONSE DOCUMENTAIRE LIMITÉE :", rejection)
        print("OUTILS UTILISÉS : aucun (blocage déterministe avant appel API)")
        return
    import os
    from dotenv import load_dotenv
    load_dotenv(DATA.parent / ".env")
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY absente du .env")
    graph = build_agent(chunks, vectors)
    run_guarded(graph, args.question)


if __name__ == "__main__":
    main()
