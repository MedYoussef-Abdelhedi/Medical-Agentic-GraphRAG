"""Tests unitaires purement locaux de l'étape 15."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from compare_vector_graphrag_v3 import (
    MODES, build_evidence, build_prompt, check_citations, export_review_csv,
    load_benchmark, summarize, verify_questions,
)

VECTOR = [{"chunk_id": "1_0", "passage": "A", "source_url_historical": "https://example.org"},
          {"chunk_id": "1_1", "passage": "B", "source_url_historical": "https://example.org"}]
GRAPH = [{"candidate_id": "C001", "disease": "Acromegaly", "symptom": "Headaches",
          "evidence_chunk_id": "1_1", "evidence_phrase": "headaches"}]


class ComparisonTests(unittest.TestCase):
    def test_benchmark_cases(self):
        examples = load_benchmark(Path(__file__).parent / "benchmark_questions_v1.json")
        self.assertEqual(len(examples), 4)

    def test_duplicate_questions_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            p.write_text('{"examples":[{"id":"x","question":"a","disease":"b","category":"supported"},'
                         '{"id":"x","question":"a","disease":"b","category":"supported"}]}')
            with self.assertRaises(ValueError): load_benchmark(p)

    def test_same_vector_context(self):
        a, _ = build_evidence("vector", VECTOR, GRAPH)
        b, _ = build_evidence("hybrid", VECTOR, GRAPH)
        self.assertEqual(a["VECTOR_PASSAGES"], b["VECTOR_PASSAGES"])
        self.assertEqual(a["GRAPH_FACTS"], [])
        self.assertEqual(len(b["GRAPH_FACTS"]), 1)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError): build_evidence("graphonly", VECTOR, GRAPH)

    def test_requires_graph_for_hybrid(self):
        with self.assertRaises(ValueError): build_evidence("hybrid", VECTOR, [])

    def test_unknown_citation_detected(self):
        _, allowed = build_evidence("vector", VECTOR, GRAPH)
        check = check_citations("Answer [999_1]", allowed, GRAPH, "vector")
        self.assertFalse(check["structurally_valid"])
        self.assertEqual(check["unknown_citations"], ["999_1"])

    def test_vector_citation_valid(self):
        _, allowed = build_evidence("vector", VECTOR, GRAPH)
        self.assertTrue(check_citations("Text [1_0]", allowed, GRAPH, "vector")["structurally_valid"])

    def test_hybrid_requires_graph_evidence_citation(self):
        _, allowed = build_evidence("hybrid", VECTOR, GRAPH)
        self.assertFalse(check_citations("Text [1_0]", allowed, GRAPH, "hybrid")["structurally_valid"])
        self.assertTrue(check_citations("Text [1_1]", allowed, GRAPH, "hybrid")["structurally_valid"])

    def test_prompt_same_structure(self):
        a, _ = build_evidence("vector", VECTOR, GRAPH)
        b, _ = build_evidence("hybrid", VECTOR, GRAPH)
        self.assertIn('"GRAPH_FACTS": []', build_prompt("Q", a))
        self.assertIn('"GRAPH_FACTS": [', build_prompt("Q", b))

    def test_invalid_router_rejected(self):
        fake = SimpleNamespace(canonical_disease=lambda _: None,
                               symptoms_intent=lambda _: True, preflight_block=lambda _: None)
        with self.assertRaises(ValueError):
            verify_questions([{"id":"X", "question":"Q", "disease":"Acromegaly"}], fake)

    def test_summary_failed_not_counted(self):
        rows = [
            {"mode":"vector", "error":None, "generation_latency_s":1, "total_latency_s":2,
             "citation_check":{"structurally_valid":True}},
            {"mode":"vector", "error":"connect_error", "generation_latency_s":1, "total_latency_s":2},
            {"mode":"hybrid", "error":None, "generation_latency_s":3, "total_latency_s":4,
             "citation_check":{"structurally_valid":True}},
        ]
        stats = summarize(rows)
        self.assertEqual(stats["vector"]["successful_generations"], 1)
        self.assertEqual(stats["vector"]["attempts"], 2)
        self.assertEqual(stats["hybrid"]["mean_end_to_end_latency_s"], 4)

    def test_manual_review_pending(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "review.csv"
            export_review_csv(p, [{"case_id":"Q", "mode":"vector", "question":"Q",
                                   "disease":"Acromegaly", "answer":"A", "error":None}])
            self.assertIn("pending", p.read_text(encoding="utf-8-sig"))


if __name__ == "__main__": unittest.main()
