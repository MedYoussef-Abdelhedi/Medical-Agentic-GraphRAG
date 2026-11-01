"""Tests offline de la validation des sorties LangGraph V3 (sans Groq)."""
import json
import unittest
from types import SimpleNamespace
from medical_graphrag.agents.agent_langgraph_v3_guarded import classify_question, preflight_block, validate_execution


def tool(name, content):
    return SimpleNamespace(type="tool", name=name, content=content)


def ai(text, tool_calls=None):
    return SimpleNamespace(type="ai", content=text, tool_calls=tool_calls or [])


def result(messages):
    return {"messages": messages}


def retrieved():
    return tool("rechercher_medquad", json.dumps({"results": [
        {"chunk_id": "1_0", "source_url_historical": "https://example.org/archive"},
        {"chunk_id": "1_1", "source_url_historical": "https://example.org/archive"},
    ]}))


class GuardedAgentTests(unittest.TestCase):
    def test_stats_question(self):
        question = "How many documents and chunks are indexed?"
        self.assertEqual(classify_question(question), "statistics")
        check = validate_execution(question, result([tool("statistiques_corpus", "{}"), ai("200 documents, 750 chunks")]))
        self.assertTrue(check["allowed"])

    def test_medical_citations_normal(self):
        check = validate_execution("What are the symptoms of acromegaly?", result([retrieved(), ai("It can affect ring size [1_0].")]))
        self.assertTrue(check["allowed"])
        self.assertEqual(check["sources"][0][0], "1_0")

    def test_medical_citations_unicode(self):
        check = validate_execution("What are the symptoms of acromegaly?", result([retrieved(), ai("It can affect ring size【1_0】.")]))
        self.assertTrue(check["allowed"])

    def test_invalid_citation(self):
        check = validate_execution("What are the symptoms of acromegaly?", result([retrieved(), ai("Invented citation [99_2].")]))
        self.assertFalse(check["allowed"])
        self.assertIn("unknown_citations", check["reason"])

    def test_no_citation(self):
        check = validate_execution("What are the symptoms of acromegaly?", result([retrieved(), ai("Uncited assertion")]))
        self.assertEqual(check["reason"], "no_chunk_citation_in_answer")

    def test_wrong_tool_stats_for_medical(self):
        check = validate_execution("What are the symptoms of acromegaly?", result([tool("statistiques_corpus", "{}"), ai("200")]))
        self.assertEqual(check["reason"], "required_tool_missing:rechercher_medquad")

    def test_invalid_trace(self):
        check = validate_execution("What are the symptoms of acromegaly?", result([tool("rechercher_medquad", "malformed"), ai("[1_0]")]))
        self.assertEqual(check["reason"], "invalid_retrieval_trace")

    def test_current_2026_guidelines_blocked(self):
        self.assertEqual(classify_question("What do the latest 2026 clinical guidelines recommend for acromegaly?"), "current_clinical")
        self.assertIsNotNone(preflight_block("What do the latest 2026 clinical guidelines recommend for acromegaly?"))

    def test_current_french_blocked(self):
        self.assertIsNotNone(preflight_block("Quelles sont les recommandations cliniques actuelles pour l'acromégalie ?"))

    def test_personal_medical_blocked(self):
        self.assertIsNotNone(preflight_block("I have symptoms, should I take medication?"))

    def test_no_tool(self):
        check = validate_execution("What are the symptoms of acromegaly?", result([ai("I think...")]))
        self.assertEqual(check["reason"], "required_tool_missing:rechercher_medquad")

    def test_unfinished_agent(self):
        check = validate_execution("What are the symptoms of acromegaly?", result([retrieved()]))
        self.assertEqual(check["reason"], "no_final_ai_message")


if __name__ == "__main__":
    unittest.main()
