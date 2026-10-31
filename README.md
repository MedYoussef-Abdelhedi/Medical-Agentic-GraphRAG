# Medical Agentic GraphRAG

A biomedical research assistant combining **Agentic RAG, GraphRAG, LangGraph, Neo4j, MCP and PubMed** for source-grounded medical information retrieval.

## Architecture

- **Vector RAG:** MedQuAD V3, Sentence-Transformers, semantic search.
- **GraphRAG:** Neo4j knowledge graph with documentary clinical relations.
- **Agent Orchestration:** LangGraph and Groq LLM.
- **MCP Integration:** Standardized access to retrieval tools.
- **Live Literature Search:** PubMed/NCBI API.
- **Evaluation:** Retrieval benchmarks, RAGAS experiments, citation checks and regression tests.

## Project Results

- 200 MedQuAD documents.
- 750 indexed chunks with 384-dimensional embeddings.
- 14 pilot disease–symptom relations in Neo4j.
- 4 MCP tools.
- Vector RAG vs GraphRAG experimental comparison.

## Main Technologies

Python · PyTorch · Sentence-Transformers · LangChain · LangGraph · Neo4j · Cypher · MCP · Groq · PubMed · RAGAS

## Source Code

The main implementation is available in `2eme_agent/`.

Principal entry point:

`python agent_graphrag_mcp_web_v3.py --question "What are the latest clinical guidelines for treating acromegaly?"`

Requires installation of the Python dependencies, configuration of the environment variables and preparation of the local data index.

## Limitations

This is a research prototype, not a clinically validated medical decision-support system. Historical MedQuAD sources are distinguished from current PubMed-indexed literature. PubMed abstracts do not necessarily represent the latest official clinical guidelines.

## Author

Mohamed Youssef Abdelhedi