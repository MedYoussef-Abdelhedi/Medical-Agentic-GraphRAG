"""Tests sans Neo4j/Groq/HF : requêtes sûres, alignement, citations, routage."""
import hashlib
import importlib.util
import unittest

from agent_graphrag_hybride_v3 import (
    EXPECTED_CORPUS_SHA256,
    QUERY_GRAPH_EVIDENCE,
    build_workflow,
    canonical_disease,
    lookup_graph,
    symptoms_intent,
    validate_answer,
    validate_graph_rows,
)
from agent_langgraph_v3_guarded import preflight_block


class FakeDriver:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def execute_query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.rows, None, None


def fixture():
    content = "Acromegaly symptoms may include sleep apnea."
    checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
    chunk = {"chunk_id": "1_1", "document_id": 1, "content": content,
             "source_url": "https://example.invalid/historical-source"}
    rows = []
    for i in range(8):
        rows.append({
            "candidate_id": f"C{i + 1:03d}", "evidence_phrase": "sleep apnea",
            "claim_chunk_id": "1_1", "claim_chunk_sha256": checksum,
            "claim_source_url": chunk["source_url"],
            "reviewer": "documentary_review", "disease": "Acromegaly",
            "symptom": f"Symptom {i + 1}", "chunk_id": "1_1",
            "chunk_content": content, "chunk_sha256": checksum,
            "source_url": chunk["source_url"], "document_id": 1,
            "source_url_from_node": chunk["source_url"],
        })
    return rows, {"1_1": chunk}


class GraphRetrievalTests(unittest.TestCase):
    def test_known_diseases_fr_en(self):
        self.assertEqual(canonical_disease("les symptômes de l'acromégalie"), "Acromegaly")
        self.assertEqual(canonical_disease("adrenal insufficiency signs"), "Adrenal insufficiency")
        self.assertEqual(canonical_disease("insuffisance surrénalienne"), "Adrenal insufficiency")
        self.assertIsNone(canonical_disease("hypertension"))

    def test_symptom_intent(self):
        self.assertTrue(symptoms_intent("Quels sont les symptômes de l'acromégalie ?"))
        self.assertTrue(symptoms_intent("Symptoms of acromegaly"))
        self.assertFalse(symptoms_intent("What is acromegaly?"))

    def test_clinical_guard_remains(self):
        self.assertTrue(preflight_block("What do the latest 2026 clinical guidelines recommend for acromegaly?"))

    def test_query_is_fixed_and_read_only(self):
        normalized = QUERY_GRAPH_EVIDENCE.upper()
        for forbidden in ("CREATE ", "MERGE ", "DELETE ", "DETACH ", " SET ", "REMOVE ", "LOAD CSV"):
            self.assertNotIn(forbidden, normalized)
        self.assertIn("$DISEASE", normalized)
        self.assertIn("$CORPUS_SHA256", normalized)
        for edge in (":SUBJECT", ":OBJECT", ":SUPPORTED_BY", ":HAS_SYMPTOM", ":HAS_CHUNK", ":FROM_SOURCE"):
            self.assertIn(edge, normalized)

    def test_graph_rows_verified(self):
        rows, by_id = fixture()
        result = validate_graph_rows(rows, "Acromegaly", by_id)
        self.assertEqual(len(result), 8)
        self.assertEqual(result[0]["evidence_chunk_id"], "1_1")

    def test_incomplete_graph_is_rejected(self):
        rows, by_id = fixture()
        with self.assertRaisesRegex(ValueError, "incomplet"):
            validate_graph_rows(rows[:7], "Acromegaly", by_id)

    def test_modified_chunk_content_is_rejected(self):
        rows, by_id = fixture()
        rows[2]["chunk_content"] = "tampered"
        with self.assertRaisesRegex(ValueError, "Provenance"):
            validate_graph_rows(rows, "Acromegaly", by_id)

    def test_mismatched_claim_document_is_rejected(self):
        rows, by_id = fixture()
        rows[0]["document_id"] = 9
        with self.assertRaisesRegex(ValueError, "Provenance"):
            validate_graph_rows(rows, "Acromegaly", by_id)

    def test_duplicate_claim_is_rejected(self):
        rows, by_id = fixture()
        rows[1]["candidate_id"] = rows[0]["candidate_id"]
        with self.assertRaisesRegex(ValueError, "dupliqué"):
            validate_graph_rows(rows, "Acromegaly", by_id)

    def test_missing_evidence_phrase_is_rejected(self):
        rows, by_id = fixture()
        rows[0]["evidence_phrase"] = "something unsupported"
        with self.assertRaisesRegex(ValueError, "Phrase"):
            validate_graph_rows(rows, "Acromegaly", by_id)

    def test_fixed_query_parameter_binding_and_read_route(self):
        rows, by_id = fixture()
        driver = FakeDriver(rows)
        result = lookup_graph(driver, "ba79ec20", "Acromegaly", by_id)
        self.assertEqual(len(result), 8)
        query, kwargs = driver.calls[0]
        self.assertIs(query, QUERY_GRAPH_EVIDENCE)
        self.assertEqual(kwargs["routing_"], "r")
        self.assertEqual(kwargs["database_"], "ba79ec20")
        self.assertEqual(kwargs["parameters_"], {"disease": "Acromegaly", "corpus_sha256": EXPECTED_CORPUS_SHA256})

    def test_valid_and_unknown_citations(self):
        facts = [{"evidence_chunk_id": "1_1"}]
        vectors = [{"chunk_id": "1_0"}]
        self.assertEqual(validate_answer("A [1_1] and [1_0]", facts, vectors), ["1_0", "1_1"])
        with self.assertRaisesRegex(ValueError, "non fournis"):
            validate_answer("A [500_99]", facts, vectors)
        with self.assertRaisesRegex(ValueError, "aucun fait"):
            validate_answer("A [1_0]", facts, vectors)
        with self.assertRaisesRegex(ValueError, "Aucune référence"):
            validate_answer("No citation", facts, vectors)

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph non installé dans l'environnement de test")
    def test_workflow_calls_both_sources_then_synthesizer(self):
        called = []
        facts = [{"evidence_chunk_id": "1_1", "symptom": "Sleep apnea"}]
        passages = [{"chunk_id": "1_0", "passage": "Historical passage"}]
        def graph_lookup(disease):
            called.append("graph")
            return facts
        def vector_lookup(question):
            called.append("vector")
            return passages
        def synthesize(question, graph, vector):
            called.append("groq")
            return "Historical symptoms [1_1]."
        app = build_workflow(graph_lookup, vector_lookup, synthesize)
        final = app.invoke({"question": "symptoms of acromegaly", "disease": "Acromegaly"})
        self.assertEqual(called, ["graph", "vector", "groq"])
        self.assertEqual(final["cited_chunks"], ["1_1"])


if __name__ == "__main__":
    unittest.main()
