"""Offline contract tests. No Neo4j, Groq, HuggingFace, or MCP SDK needed."""
from __future__ import annotations

import asyncio
import json
import types
import unittest
from unittest.mock import patch

import numpy as np
import mcp_medquad_service_v3 as service
from mcp_medquad_client_v3 import extract_structured, EXPECTED_TOOLS
from agent_graphrag_mcp_v3 import validate_tool_data


class FakeHybrid:
    BASE = __import__('pathlib').Path(__file__).resolve().parents[1] / 'src' / 'medical_graphrag'
    DATA = BASE / 'data'

    def __init__(self):
        self.calls = []
        self.vectors = np.ones((2, 384), dtype=np.float32)
        self.rows = [{'chunk_id': '1_0', 'document_id': 1}, {'chunk_id': '2_0', 'document_id': 2}]

    def load_verified_index(self, data):
        self.calls.append(('load', str(data)))
        return self.rows, self.vectors

    def check_corpus(self, rows):
        self.calls.append(('check', len(rows)))
        return {r['chunk_id']: r for r in rows}

    def preflight_block(self, question):
        return 'Historic corpus cannot validate current guidance' if '2026' in question else None

    def semantic_search(self, question, chunks, vectors, top_k=3):
        self.calls.append(('semantic', question, top_k))
        return [{'chunk_id': '1_0', 'passage': 'Example source passage', 'source_url_historical': 'http://example.org'}]

    def canonical_disease(self, question):
        return 'Acromegaly' if 'acro' in question.lower() else None

    def lookup_graph_live(self, disease, by_id):
        self.calls.append(('graph', disease))
        return [{'candidate_id': 'C001', 'disease': disease, 'symptom': 'Headaches',
                 'evidence_chunk_id': '1_0', 'evidence_phrase': 'headaches'}]


class ServiceTests(unittest.TestCase):
    def setUp(self):
        service._verified_index.cache_clear()
        self.fake = FakeHybrid()
        self.patch = patch.object(service, '_hybrid', return_value=self.fake)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        service._verified_index.cache_clear()

    def test_stats_no_external_calls(self):
        result = service.corpus_stats()
        self.assertEqual((result['documents'], result['chunks'], result['vector_dimensions']), (2, 2, 384))
        self.assertFalse(any(x[0] in {'semantic', 'graph'} for x in self.fake.calls))

    def test_reuse_index(self):
        service.corpus_stats(); service.corpus_stats()
        self.assertEqual(sum(x[0] == 'load' for x in self.fake.calls), 1)

    def test_vector_search(self):
        data = service.vector_search('Symptoms of acromegaly?', 1)
        self.assertEqual(data['results'][0]['chunk_id'], '1_0')
        self.assertEqual(self.fake.calls[-1], ('semantic', 'Symptoms of acromegaly?', 1))

    def test_vector_guard(self):
        data = service.vector_search('Latest 2026 recommendations')
        self.assertEqual(data['status'], 'not_supported')
        self.assertFalse(any(x[0] == 'semantic' for x in self.fake.calls))

    def test_empty_input(self):
        with self.assertRaises(ValueError): service.vector_search('   ')

    def test_too_long(self):
        with self.assertRaises(ValueError): service.vector_search('a' * 501)

    def test_top_k_limit(self):
        for value in (0, 6, -1, True, '3'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                service.vector_search('Symptoms?', value)

    def test_known_graph(self):
        data = service.graph_symptoms('Acromegaly')
        self.assertEqual(data['claims'][0]['candidate_id'], 'C001')
        self.assertIn(('graph', 'Acromegaly'), self.fake.calls)

    def test_unknown_graph_is_not_supported(self):
        data = service.graph_symptoms('Unknown disease')
        self.assertEqual(data['status'], 'not_supported')
        self.assertFalse(any(x[0] == 'graph' for x in self.fake.calls))

    def test_empty_disease(self):
        with self.assertRaises(ValueError): service.graph_symptoms('')


class ProtocolContractTests(unittest.TestCase):
    def test_expected_names(self):
        self.assertEqual(len(EXPECTED_TOOLS), 3)

    def test_extract_structured(self):
        result = types.SimpleNamespace(is_error=False, structured_content={'status':'ok', 'results':[]})
        self.assertEqual(extract_structured(result)['status'], 'ok')

    def test_reject_mcp_error(self):
        result = types.SimpleNamespace(is_error=True, structured_content={'status':'ok'}, content=[])
        with self.assertRaises(RuntimeError): extract_structured(result)

    def test_reject_unstructured(self):
        result = types.SimpleNamespace(is_error=False, structured_content=None)
        with self.assertRaises(ValueError): extract_structured(result)

    def test_validate_empty_evidence(self):
        with self.assertRaises(ValueError): validate_tool_data({'status':'ok', 'claims': []}, 'claims')

    def test_validate_blocked(self):
        with self.assertRaises(ValueError): validate_tool_data({'status':'not_supported', 'reason':'not covered'}, 'results')

    def test_validate_evidence(self):
        self.assertEqual(validate_tool_data({'status': 'ok', 'results':[{'chunk_id':'1_0'}]},'results')[0]['chunk_id'],'1_0')


class MCPToolResultCompatibilityTests(unittest.TestCase):
    """No MCP installation necessary; these are tool envelope shape contracts."""

    @staticmethod
    def _result(*, structured=None, content=None, is_error=False):
        return types.SimpleNamespace(
            structured_content=structured,
            content=content if content is not None else [],
            is_error=is_error,
        )

    @staticmethod
    def _text(data):
        return types.SimpleNamespace(type='text', text=data)

    def test_direct_structured_object(self):
        obj = {'status': 'ok', 'documents': 200, 'chunks': 750}
        self.assertEqual(extract_structured(self._result(structured=obj)), obj)

    def test_single_result_envelope(self):
        obj = {'status': 'ok', 'chunks': 750}
        self.assertEqual(extract_structured(self._result(structured={'result': obj})), obj)

    def test_string_json_in_result_envelope(self):
        obj = {'status': 'ok', 'chunks': 750}
        self.assertEqual(extract_structured(self._result(structured={'result': json.dumps(obj)})), obj)

    def test_valid_json_text_fallback(self):
        obj = {'status': 'ok', 'chunks': 750}
        res = self._result(content=[self._text(json.dumps(obj))])
        self.assertEqual(extract_structured(res), obj)

    def test_text_json_result_envelope(self):
        obj = {'status': 'ok', 'chunks': 750}
        res = self._result(content=[self._text(json.dumps({'result': obj}))])
        self.assertEqual(extract_structured(res), obj)

    def test_mcp_tool_error_rejected(self):
        res = self._result(structured={'status': 'ok'}, is_error=True, content=[self._text('failure')])
        with self.assertRaises(RuntimeError):
            extract_structured(res)

    def test_unstructured_prose_rejected(self):
        res = self._result(content=[self._text('Everything is okay, 750 chunks')])
        with self.assertRaises(ValueError):
            extract_structured(res)

    def test_inconsistent_dual_channels_rejected(self):
        res = self._result(structured={'status': 'ok', 'chunks': 750},
                           content=[self._text(json.dumps({'status': 'ok', 'chunks': 740}))])
        with self.assertRaises(ValueError):
            extract_structured(res)

    def test_unknown_status_rejected(self):
        res = self._result(structured={'status': 'hallucinated', 'chunks': 750})
        with self.assertRaises(ValueError):
            extract_structured(res)


if __name__ == '__main__':
    unittest.main()

# Optional REAL MCP SDK contract smoke test. Skipped on machines without MCP 2.x;
# user's environment should run it after pip install.
import importlib.util

@unittest.skipUnless(importlib.util.find_spec('mcp'), 'MCP Python SDK not installed')
class MCP2InProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_protocol_list_and_stats(self):
        from mcp import Client
        import mcp_medquad_server_v3 as server
        with patch.object(server, 'corpus_stats', return_value={
            'status': 'ok', 'documents': 200, 'chunks': 750, 'index_version': 'V3'
        }):
            async with Client(server.mcp, raise_exceptions=True) as client:
                result = await client.list_tools()
                self.assertEqual({x.name for x in result.tools}, EXPECTED_TOOLS)
                res = await client.call_tool('statistiques_medquad_v3', {})
                self.assertEqual(extract_structured(res)['chunks'], 750)
