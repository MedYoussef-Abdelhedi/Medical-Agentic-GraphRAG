"""Tests unitaires hors ligne pour l'import du graphe documentaire MedQuAD V3."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from import_medquad_v3_neo4j import (
    CONSTRAINTS, READ_COUNTS, WRITE_CHUNKS, WRITE_DOCUMENTS, WRITE_SOURCES,
    READ_PROVENANCE_SOURCES, READ_PROVENANCE_DOCUMENTS, READ_PROVENANCE_CHUNKS,
    batches, import_graph, load_payload, make_source_key, verify,
)


class FakeDriver:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def execute_query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if query == READ_PROVENANCE_SOURCES:
            return (list(self.payload.sources), None, [])
        if query == READ_PROVENANCE_DOCUMENTS:
            return ([{**{k:v for k,v in item.items() if k != 'source_key'},
                      'source_keys':[item['source_key']]}
                     for item in self.payload.documents], None, [])
        if query == READ_PROVENANCE_CHUNKS:
            return ([{**{k:v for k,v in item.items() if k != 'document_key'},
                      'document_keys':[item['document_key']]}
                     for item in self.payload.chunks], None, [])
        for category, cypher in READ_COUNTS.items():
            if query == cypher:
                counts = self.payload.counts()
                val = counts.get(category, 1 if category == 'dated_docs' else 0)
                return ([{'n': val}], None, [])
        return ([], None, [])


class ImportTests(unittest.TestCase):
    def build_fixture(self, root: Path):
        data = root / 'data'
        data.mkdir()
        records = []
        for d in range(200):
            for j in range(4 if d < 150 else 3):
                cid = f'{d}_{j}'
                source_url = f'https://medical.example/source/{d % 29}'
                records.append({
                    'document_id': d, 'chunk_id': cid,
                    'question': f'What is test {d}?', 'content': f'Content for document {d} chunk {j}.',
                    'source_file': f'file-{d%29}.xml', 'source_url': source_url,
                    'text_for_embedding': f'Question: What is test {d}?\nAnswer: Content for document {d} chunk {j}.',
                    **({'source_date_text': 'February 2014', 'source_date_evidence_chunk_id': '188_10'} if d == 188 else {}),
                })
        self.assertEqual(len(records), 750)
        chunk_bytes = (json.dumps(records, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
        (data / 'medquad_chunks_v3.json').write_bytes(chunk_bytes)
        embedding_bytes = b'fake_vectors_for_checksum_fixture'
        (data / 'medquad_embeddings_v3.npy').write_bytes(embedding_bytes)
        meta = {
            'chunks_sha256': hashlib.sha256(chunk_bytes).hexdigest(),
            'embeddings_sha256': hashlib.sha256(embedding_bytes).hexdigest(),
            'chunks_after': 750, 'source_documents': 200,
        }
        (data / 'index_manifest_v3.json').write_text(json.dumps(meta), encoding='utf8')
        return data, meta['chunks_sha256']

    def test_prepare_reproducible_payload(self):
        with tempfile.TemporaryDirectory() as t:
            data, sha = self.build_fixture(Path(t))
            p = load_payload(data, expected_sha=sha)
            self.assertEqual(p.counts(), {'sources':29, 'documents':200,'chunks':750,'source_links':200,'chunk_links':750})
            self.assertEqual(p.documents[188]['source_date_text'], 'February 2014')
            self.assertEqual(p.chunks[0]['key'], 'medquad-v3:chunk:0_0')

    def test_manifest_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            data, sha = self.build_fixture(Path(t))
            with (data / 'medquad_chunks_v3.json').open('ab') as handle:
                handle.write(b' ')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load_payload(data, expected_sha=sha)

    def test_expected_corpus_digest_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            data, sha = self.build_fixture(Path(t))
            with self.assertRaisesRegex(ValueError, 'Corpus inattendu'):
                load_payload(data)

    def test_duplicate_chunk_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            data, sha = self.build_fixture(Path(t))
            p = data / 'medquad_chunks_v3.json'
            rows = json.loads(p.read_text(encoding='utf8'))
            rows[1]['chunk_id'] = rows[0]['chunk_id']
            p.write_text(json.dumps(rows), encoding='utf8')
            new_sha = hashlib.sha256(p.read_bytes()).hexdigest()
            manifest = json.loads((data / 'index_manifest_v3.json').read_text())
            manifest['chunks_sha256'] = new_sha
            (data / 'index_manifest_v3.json').write_text(json.dumps(manifest), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'duplique'):
                load_payload(data, expected_sha=new_sha)

    def test_zero_destructive_cypher(self):
        statements = CONSTRAINTS + [WRITE_SOURCES, WRITE_DOCUMENTS, WRITE_CHUNKS]
        for statement in statements:
            self.assertNotIn('DELETE ', statement.upper())
            self.assertNotIn('DROP ', statement.upper())
        self.assertTrue(all('MERGE ' in stmt for stmt in statements[3:]))

    def test_batched_idempotent_writes(self):
        with tempfile.TemporaryDirectory() as t:
            data, sha = self.build_fixture(Path(t))
            p = load_payload(data, expected_sha=sha)
            fake = FakeDriver(p)
            with patch('builtins.print'):
                import_graph(fake, 'neo4j', p)
            self.assertEqual(len(fake.calls), 3 + 1 + 2 + 8)
            self.assertEqual(fake.calls[-1][1]['parameters_']['import_sha256'], sha)
            self.assertEqual(fake.calls[-1][1]['database_'], 'neo4j')
            self.assertEqual(verify(fake,'neo4j',p)['chunks'],750)

    def test_bad_graph_counts_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            data, sha = self.build_fixture(Path(t))
            p = load_payload(data, expected_sha=sha)
            fake = FakeDriver(p)
            orig = fake.execute_query
            def bad(query, **kwargs):
                if query == READ_COUNTS['chunks']:
                    return ([{'n': 749}],None,[])
                return orig(query, **kwargs)
            fake.execute_query = bad
            with self.assertRaisesRegex(RuntimeError, 'Verification Neo4j non conforme'):
                verify(fake,'neo4j',p)

    def test_swapped_chunk_link_is_detected(self):
        with tempfile.TemporaryDirectory() as t:
            data, sha = self.build_fixture(Path(t))
            p = load_payload(data, expected_sha=sha)
            fake = FakeDriver(p)
            orig = fake.execute_query
            def corrupt_link(query, **kwargs):
                result = orig(query, **kwargs)
                if query == READ_PROVENANCE_CHUNKS:
                    rows, _, _ = result
                    rows[0]['document_keys'] = ['medquad-v3:doc:77']
                    return rows, None, []
                return result
            fake.execute_query = corrupt_link
            with self.assertRaisesRegex(RuntimeError, 'Lien de provenance incorrect'):
                verify(fake, 'neo4j', p)

    def test_corrupted_chunk_content_is_detected(self):
        with tempfile.TemporaryDirectory() as t:
            data, sha = self.build_fixture(Path(t))
            p = load_payload(data, expected_sha=sha)
            fake = FakeDriver(p)
            orig = fake.execute_query
            def corrupt_content(query, **kwargs):
                result = orig(query, **kwargs)
                if query == READ_PROVENANCE_CHUNKS:
                    rows, _, _ = result
                    rows[0]['content'] = 'changed'
                    return rows, None, []
                return result
            fake.execute_query = corrupt_content
            with self.assertRaisesRegex(RuntimeError, 'Propriete Neo4j invalide'):
                verify(fake, 'neo4j', p)

    def test_source_keys_stable(self):
        self.assertEqual(make_source_key('https://example.test'), make_source_key('https://example.test'))
        self.assertNotEqual(make_source_key('https://example.test'), make_source_key('https://example.test/'))


if __name__ == '__main__':
    unittest.main()
