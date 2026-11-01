"""Offline tests for optional PubMed MCP tool, no web calls."""
from __future__ import annotations
import json
import types
import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch
from medical_graphrag.mcp_tools.mcp_medquad_web_service_v1 import parse_pubmed_xml, search_pubmed_guidelines
from medical_graphrag.agents.agent_graphrag_mcp_web_v3 import needs_live_guidelines, verified_pubmed_articles, web_synthesize
from medical_graphrag.mcp_tools.mcp_medquad_web_client_v3 import EXPECTED_WEB_TOOLS

XML = b'''<?xml version="1.0" encoding="utf-8"?><PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>123456</PMID><Article>
<Journal><JournalIssue><PubDate><Year>2025</Year><Month>Oct</Month></PubDate></JournalIssue></Journal>
<ArticleTitle>Acromegaly clinical practice guideline</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">This consensus discusses a historical set of diagnostic and treatment approaches for acromegaly; details are incomplete and clinical applicability must be independently checked.</AbstractText></Abstract>
<PublicationTypeList><PublicationType>Practice Guideline</PublicationType></PublicationTypeList>
</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>'''

class WebToolTests(unittest.TestCase):
    def test_extract_article(self):
        x = parse_pubmed_xml(XML)[0]
        self.assertEqual(x['pmid'], '123456')
        self.assertEqual(x['publication_date'], '2025-Oct')
        self.assertEqual(x['url'], 'https://pubmed.ncbi.nlm.nih.gov/123456/')
        self.assertTrue(x['has_abstract'])
        self.assertIn('Practice Guideline', x['publication_types'])
    def test_two_fixed_ncbi_endpoints(self):
        urls = []
        def fake(url):
            urls.append(url)
            if 'esearch.fcgi' in url: return b'{"esearchresult":{"idlist":["123456"]}}'
            if 'efetch.fcgi' in url: return XML
            raise AssertionError('Unexpected endpoint')
        with patch.dict('os.environ', {}, clear=True):
            result = search_pubmed_guidelines('Acromegaly', downloader=fake)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(len(urls), 2)
        self.assertTrue(all(u.startswith('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/') for u in urls))
        self.assertIn('consensus[Title]', parse_qs(urlparse(urls[0]).query)['term'][0])
    def test_no_results_fail_closed(self):
        self.assertEqual(search_pubmed_guidelines('Acromegaly', downloader=lambda _: b'{"esearchresult":{"idlist":[]}}')['status'], 'not_supported')
    def test_bad_response_fail_closed(self):
        self.assertEqual(search_pubmed_guidelines('Acromegaly', downloader=lambda _: b'<bad')['status'], 'not_supported')
    def test_bad_input(self):
        for topic in ('', 'a', 'x'*101, 'https://invalid/url', 'x?x=3'):
            with self.subTest(topic=topic), self.assertRaises(ValueError): search_pubmed_guidelines(topic)
    def test_max_results(self):
        for k in (0, 9, '2', True):
            with self.subTest(k=k), self.assertRaises(ValueError): search_pubmed_guidelines('Acromegaly', max_results=k)
    def test_bad_identifier_doesnt_fetch(self):
        self.assertEqual(search_pubmed_guidelines('Acromegaly', downloader=lambda _: b'{"esearchresult":{"idlist":["../etc/passwd"]}}')['status'], 'not_supported')

class RoutingTests(unittest.TestCase):
    def test_current_2026(self): self.assertTrue(needs_live_guidelines('What are the latest 2026 clinical guidelines for treating acromegaly?'))
    def test_current_french(self): self.assertTrue(needs_live_guidelines('Quelles sont les recommandations actuelles de traitement de l’acromégalie ?'))
    def test_old_symptom_routed_to_rag(self): self.assertFalse(needs_live_guidelines("Quels sont les symptômes de l'acromégalie ?"))
    def test_four_tools(self): self.assertEqual(len(EXPECTED_WEB_TOOLS), 4)
    def test_invalid_pmid(self):
        with self.assertRaises(ValueError): verified_pubmed_articles({'status':'ok','results':[{'pmid':'fake','title':'X','url':'bad'}]})
    def test_no_results(self): self.assertEqual(verified_pubmed_articles({'status':'not_supported','results':[]}), [])
    def test_url_mismatch(self):
        with self.assertRaises(ValueError): verified_pubmed_articles({'status':'ok','results':[{'pmid':'123456','title':'X','url':'https://pubmed.ncbi.nlm.nih.gov/999999/'}]})

class SynthTests(unittest.TestCase):
    class FakeModel:
        def __init__(self, text): self.text = text
        def invoke(self, _): return types.SimpleNamespace(content=self.text, response_metadata={'finish_reason':'stop'})
    def test_valid_citation(self):
        s = 'The source is indexed in 2025 [PMID:123456].\n[[END_RESPONSE]]'
        answer, tries = web_synthesize('Latest guidelines?', parse_pubmed_xml(XML), llm_factory=lambda _: self.FakeModel(s))
        self.assertIn('PMID:123456', answer)
        self.assertTrue(tries[0]['citation_ids_valid'])
    def test_invalid_citation(self):
        s = 'Incorrect source [PMID:999999].\n[[END_RESPONSE]]'
        with self.assertRaises(ValueError): web_synthesize('Latest guidelines?', parse_pubmed_xml(XML), llm_factory=lambda _: self.FakeModel(s))
    def test_no_abstract(self):
        rows = parse_pubmed_xml(XML)
        rows[0]['abstract'] = ''
        with self.assertRaises(ValueError): web_synthesize('Latest guidelines?', rows, llm_factory=lambda _: self.FakeModel('hi'))
    def test_incomplete_response(self):
        with self.assertRaises(ValueError): web_synthesize('Latest guidelines?', parse_pubmed_xml(XML), llm_factory=lambda _: self.FakeModel('unfinished'))

if __name__ == '__main__': unittest.main()
