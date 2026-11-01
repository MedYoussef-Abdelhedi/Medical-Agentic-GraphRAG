"""LangGraph -> MCP stdio client -> validated V3 Vector + Neo4j -> Groq.

The graph tools execute ONLY through MCP (not through direct Python calls).
No modification of index or Neo4j. Groq is used only for --question.

    python agent_graphrag_mcp_v3.py --check
    python agent_graphrag_mcp_v3.py --list-tools
    python agent_graphrag_mcp_v3.py --question "Quels sont les symptômes de l'acromégalie ?"
"""
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, TypedDict

from mcp_medquad_client_v3 import EXPECTED_TOOLS, extract_structured, server_parameters


class AgentState(TypedDict, total=False):
    question: str
    disease: str
    vector_evidence: list[dict]
    graph_evidence: list[dict]
    answer: str
    cited_chunks: list[str]
    generation_attempts: list[dict]


def validate_tool_data(data: dict, key: str) -> list[dict]:
    if data.get("status") != "ok":
        raise ValueError(f"MCP response not usable: {data.get('reason', data.get('status'))}")
    values = data.get(key)
    if not isinstance(values, list) or not values or not all(isinstance(x, dict) for x in values):
        raise ValueError(f"Invalid or empty MCP evidence: {key}")
    return values


def make_workflow(client: Any):
    import agent_graphrag_hybride_v3 as hybrid
    import compare_vector_graphrag_v3 as baseline
    import quality_generation_v3 as quality
    from langgraph.graph import StateGraph, START, END

    async def fetch_graph(state: AgentState):
        data = extract_structured(await client.call_tool(
            "symptomes_neo4j_v3", {"disease": state["disease"]}
        ))
        claims = validate_tool_data(data, "claims")
        if data.get("disease") != state["disease"]:
            raise ValueError("Neo4j disease mismatch")
        return {"graph_evidence": claims}

    async def fetch_vector(state: AgentState):
        data = extract_structured(await client.call_tool(
            "rechercher_passages_v3", {"question": state["question"], "top_k": 3}
        ))
        return {"vector_evidence": validate_tool_data(data, "results")}

    async def synthesize(state: AgentState):
        graph, vector = state["graph_evidence"], state["vector_evidence"]
        evidence, _ = baseline.build_evidence("hybrid", vector, graph)
        output = await asyncio.to_thread(quality.generate_quality_response, state["question"], evidence)
        cited = hybrid.validate_answer(output.answer, graph, vector)
        return {"answer": output.answer, "cited_chunks": cited,
                "generation_attempts": output.attempts}

    graph = StateGraph(AgentState)
    graph.add_node("mcp_neo4j_relations", fetch_graph)
    graph.add_node("mcp_recherche_vectorielle", fetch_vector)
    graph.add_node("synthese_groq", synthesize)
    graph.add_edge(START, "mcp_neo4j_relations")
    graph.add_edge("mcp_neo4j_relations", "mcp_recherche_vectorielle")
    graph.add_edge("mcp_recherche_vectorielle", "synthese_groq")
    graph.add_edge("synthese_groq", END)
    return graph.compile()


async def run_agent(args) -> None:
    import agent_graphrag_hybride_v3 as hybrid
    chunks, vectors = hybrid.load_verified_index(hybrid.DATA)
    hybrid.check_corpus(chunks)
    print(f"INDEX V3 OK: {len(chunks)} chunks / {len({r['document_id'] for r in chunks})} documents")
    if args.check:
        print("CHECK: PASS - no MCP, Neo4j, Hugging Face or Groq calls")
        return
    if args.question is not None:
        if not args.question.strip():
            raise ValueError("Question vide")
        if (blocked := hybrid.preflight_block(args.question)):
            print("DOCUMENTARY LIMIT:", blocked)
            print("No MCP or Groq calls")
            return
        disease = hybrid.canonical_disease(args.question)
        if not disease or not hybrid.symptoms_intent(args.question):
            print("Pilot limited to symptom questions about Acromegaly or Adrenal insufficiency.")
            return
        from dotenv import load_dotenv
        load_dotenv(hybrid.BASE / ".env")
        if not os.environ.get("GROQ_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY not found in 2eme_agent/.env")

    from mcp import Client
    async with Client(server_parameters()) as client:
        tool_list = await client.list_tools()
        actual = {x.name for x in tool_list.tools}
        if actual != EXPECTED_TOOLS:
            raise ValueError(f"Unexpected MCP tools: {actual}")
        print("MCP HANDSHAKE PASS; TOOLS:", ", ".join(sorted(actual)))
        if args.list_tools:
            return
        workflow = make_workflow(client)
        state = await workflow.ainvoke({"question": args.question, "disease": disease})
        print("\nTOOLS: MCP Neo4j + MCP Vector V3 + Groq")
        print("GRAPH CLAIM IDS:", ", ".join(x["candidate_id"] for x in state["graph_evidence"]))
        print("VECTOR CHUNK IDS:", ", ".join(x["chunk_id"] for x in state["vector_evidence"]))
        print("GENERATION:", state["generation_attempts"])
        print("\nANSWER:\n", state["answer"])
        print("CITATIONS:", ", ".join(f"[{c}]" for c in state["cited_chunks"]))
        print("LIMITATION: structural references only, not medically validated.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--list-tools", action="store_true")
    group.add_argument("--question", type=str)
    args = parser.parse_args()
    asyncio.run(run_agent(args))


if __name__ == "__main__":
    main()
