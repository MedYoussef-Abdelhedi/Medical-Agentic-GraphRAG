"""Tests locaux de l'index, sans Groq ni LangGraph (python -m unittest -v)."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from medical_graphrag.agents.agent_langgraph_v3 import MODEL, load_verified_index


class V3IntegrityTests(unittest.TestCase):
    def build_fixture(self, root: Path):
        data = root / "data"
        data.mkdir()
        chunks = [
            {"chunk_id": "1_0", "document_id": 1, "question": "Q", "content": "A"},
            {"chunk_id": "2_0", "document_id": 2, "question": "Q", "content": "B"},
        ]
        cpath = data / "medquad_chunks_v3.json"
        vpath = data / "medquad_embeddings_v3.npy"
        cpath.write_text(json.dumps(chunks), encoding="utf-8")
        matrix = np.zeros((2, 384), dtype=np.float32)
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        np.save(vpath, matrix)
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        manifest = {
            "model": MODEL, "chunks_sha256": sha(cpath), "embeddings_sha256": sha(vpath),
            "chunks_after": 2, "source_documents": 2,
        }
        (data / "index_manifest_v3.json").write_text(json.dumps(manifest), encoding="utf-8")
        return data, cpath

    def test_verified_index(self):
        with tempfile.TemporaryDirectory() as d:
            data, _ = self.build_fixture(Path(d))
            rows, vectors = load_verified_index(data)
            self.assertEqual(len(rows), 2)
            self.assertEqual(vectors.shape, (2, 384))

    def test_rejects_modified_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            data, cpath = self.build_fixture(Path(d))
            cpath.write_text(cpath.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_verified_index(data)

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            data, cpath = self.build_fixture(Path(d))
            chunks = json.loads(cpath.read_text(encoding="utf-8"))
            chunks[1]["chunk_id"] = chunks[0]["chunk_id"]
            cpath.write_text(json.dumps(chunks), encoding="utf-8")
            manifest_path = data / "index_manifest_v3.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["chunks_sha256"] = hashlib.sha256(cpath.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dupliqués"):
                load_verified_index(data)


if __name__ == "__main__":
    unittest.main()
