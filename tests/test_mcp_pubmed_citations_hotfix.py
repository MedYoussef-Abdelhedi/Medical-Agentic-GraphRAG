"""Offline regression tests for MCP PubMed citation patch; no NCBI / Groq calls."""
import types
import unittest

from medical_graphrag.agents.agent_graphrag_mcp_web_v3 import (
    directly_relevant_articles,
    normalize_pmid_citations,
    verified_pubmed_articles,
    web_synthesize,
)


def article(pmid='42050227', title='Consensus on acromegaly complications: an update.'):
    return {
        'pmid': pmid,
        'title': title,
        'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
        'publication_date': '2026-04-29',
        'publication_types': ['Consensus Statement'],
        'abstract': ('The expert consensus considers clinical management of complications '
                     'in acromegaly. The abstract discusses comorbidities and evidence '
                     'but does not establish universal recommendations for all patients.'),
    }


class FakeModel:
    def __init__(self, answer):
        self.answer = answer
        self.seen_messages = None
    def invoke(self, messages):
        self.seen_messages = messages
        return types.SimpleNamespace(content=self.answer, response_metadata={'finish_reason':'stop'})


class CitationRegressionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [article()]
    def test_exact(self):
        normalized, ids = normalize_pmid_citations('Discussion [PMID:42050227]')
        self.assertEqual(ids, ['42050227'])
        self.assertEqual(normalized, 'Discussion [PMID:42050227]')
    def test_whitespace_case(self):
        normalized, ids = normalize_pmid_citations('Discussion [ pmid : 42050227 ]')
        self.assertEqual(normalized, 'Discussion [PMID:42050227]')
        self.assertEqual(ids, ['42050227'])
    def test_parentheses(self):
        normalized, ids = normalize_pmid_citations('Discussion (PMID: 42050227)')
        self.assertEqual(normalized, 'Discussion [PMID:42050227]')
        self.assertEqual(ids, ['42050227'])
    def test_index_only_never_inferred(self):
        _, ids = normalize_pmid_citations('Discussion [1]')
        self.assertEqual(ids, [])
    def test_valid_answer_passes(self):
        text = 'A consensus publication discusses complications [PMID: 42050227].\n[[END_RESPONSE]]'
        out, attempts = web_synthesize('Recent publications?', self.rows,
                                      llm_factory=lambda _: FakeModel(text))
        self.assertIn('[PMID:42050227]', out)
        self.assertTrue(attempts[0]['citation_ids_valid'])
    def test_unknown_pmid_rejected(self):
        text = 'A guideline [PMID:999999].\n[[END_RESPONSE]]'
        with self.assertRaisesRegex(ValueError, 'unknown_pmid'):
            web_synthesize('Recent publications?', self.rows,
                           llm_factory=lambda _: FakeModel(text))
    def test_unlabeled_numeric_rejected(self):
        text = 'A consensus was published [1].\n[[END_RESPONSE]]'
        with self.assertRaisesRegex(ValueError, 'no_recognized_pmid_citation'):
            web_synthesize('Recent publications?', self.rows,
                           llm_factory=lambda _: FakeModel(text))
    def test_retry_receives_stronger_citation_prompt(self):
        models = {
            1100: FakeModel('It is a guideline [1].\n[[END_RESPONSE]]'),
            1700: FakeModel('A consensus exists [PMID:42050227].\n[[END_RESPONSE]]'),
        }
        answer, attempts = web_synthesize('Latest guidelines?', self.rows,
                                          llm_factory=lambda budget: models[budget])
        self.assertEqual(len(attempts), 2)
        self.assertFalse(attempts[0]['citation_ids_valid'])
        self.assertTrue(attempts[1]['citation_ids_valid'])
        self.assertIn('[PMID:42050227]', models[1700].seen_messages[-1][1])
        self.assertIn('[PMID:42050227]', answer)
    def test_invalid_end_marker_fails(self):
        with self.assertRaises(ValueError):
            web_synthesize('Latest guidelines?', self.rows,
                           llm_factory=lambda _: FakeModel('No marker [PMID:42050227].'))
    def test_specific_title_filter(self):
        rows = [article(), article('42776342', 'Pituitary radiotherapy consensus'),
                article('41820035', 'Guidelines for obstructive sleep apnea in adults')]
        direct = directly_relevant_articles(rows, 'Acromegaly')
        self.assertEqual([x['pmid'] for x in direct], ['42050227'])
    def test_specific_addison_synonym(self):
        self.assertEqual(len(directly_relevant_articles(
            [article(title="Management of Addison's disease")],
            'Adrenal insufficiency')) , 1)
    def test_verified_rejects_wrong_url(self):
        row = article()
        row['url'] = 'https://pubmed.ncbi.nlm.nih.gov/999999/'
        with self.assertRaises(ValueError):
            verified_pubmed_articles({'status': 'ok', 'results': [row]})


if __name__ == '__main__':
    unittest.main()
