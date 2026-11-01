"""Offline tests; no network, no Groq and no Neo4j required."""
import unittest
from types import SimpleNamespace
from medical_graphrag.evaluation.quality_generation_v3 import (
    check_response_completeness, generate_quality_response, IncompleteResponseError, SYSTEM,
)

EVIDENCE = {"VECTOR_PASSAGES": [{"chunk_id": "9_0", "passage": "fatigue", "source_url_historical": "example"}],
            "GRAPH_FACTS": []}


class Factory:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.budgets = []
    def __call__(self, budget):
        self.budgets.append(budget)
        outer = self
        class FakeLLM:
            def invoke(self, messages):
                return next(outer.answers)
        return FakeLLM()


def msg(answer, reason="stop", usage=None):
    return SimpleNamespace(content=answer, response_metadata={"finish_reason": reason, "token_usage": usage or {}})


class QualityTests(unittest.TestCase):
    def test_valid_marked_response(self):
        ok, why, content = check_response_completeness("A [9_0]\n[[END_RESPONSE]]", "stop")
        self.assertTrue(ok)
        self.assertEqual(content, "A [9_0]")

    def test_missing_marker_rejected(self):
        self.assertFalse(check_response_completeness("These symptoms can occur", "stop")[0])

    def test_length_finish_rejected_even_with_marker(self):
        self.assertFalse(check_response_completeness("A [9_0] [[END_RESPONSE]]", "length")[0])

    def test_unknown_reason_rejected(self):
        self.assertFalse(check_response_completeness("A [9_0] [[END_RESPONSE]]", "tool_calls")[0])

    def test_missing_finish_reason_rejected(self):
        self.assertFalse(check_response_completeness("A [9_0] [[END_RESPONSE]]", None)[0])

    def test_non_text_rejected(self):
        self.assertFalse(check_response_completeness(["not text"], "stop")[0])

    def test_marker_not_at_end_rejected(self):
        self.assertFalse(check_response_completeness("A [[END_RESPONSE]] extra", "stop")[0])

    def test_retry_incomplete_then_success(self):
        factory = Factory([msg("Incomplete", "length"), msg("Complete [9_0]\n[[END_RESPONSE]]", usage={"completion_tokens": 40})])
        result = generate_quality_response("Q", EVIDENCE, budgets=(500, 800), llm_factory=factory)
        self.assertEqual(factory.budgets, [500, 800])
        self.assertEqual(result.answer, "Complete [9_0]")
        self.assertFalse(result.attempts[0]["complete"])
        self.assertTrue(result.attempts[1]["complete"])
        self.assertEqual(result.attempts[1]["completion_tokens"], 40)

    def test_exhausted_retries_raise_not_return_truncated(self):
        factory = Factory([msg("bad"), msg("bad again")])
        with self.assertRaises(IncompleteResponseError) as err:
            generate_quality_response("Q", EVIDENCE, budgets=(50, 100), llm_factory=factory)
        self.assertEqual(len(err.exception.attempts), 2)

    def test_invalid_question_rejected(self):
        with self.assertRaises(ValueError): generate_quality_response("  ", EVIDENCE, llm_factory=Factory([]))

    def test_prompt_has_documentary_qualifiers(self):
        for concept in ("Addison", "most common", "GRAPH_FACTS", "VECTOR_PASSAGES", "in women"):
            self.assertIn(concept, SYSTEM)


if __name__ == "__main__": unittest.main()
