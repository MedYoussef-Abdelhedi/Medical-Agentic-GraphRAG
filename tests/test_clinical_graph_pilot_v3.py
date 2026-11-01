"""Tests purement locaux, sans Groq ni Neo4j."""
import csv
import copy
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from medical_graphrag.graph import clinical_graph_pilot_v3 as cg


def mock_payload():
    chunks = []
    for cid, doc_id, question, content in (
        ("1_0", 1, "What are the symptoms of Acromegaly ?", "One feature is the abnormal growth of the hands and feet."),
        ("1_1", 1, "What are the symptoms of Acromegaly ?", "joint aches; sleep apnea; excessive sweating; headaches; impaired vision; fatigue and weakness; thick, coarse, oily skin"),
        ("9_0", 9, "What are the symptoms of Adrenal Insufficiency and Addison's Disease ?", "chronic, or long lasting, fatigue; muscle weakness; loss of appetite; weight loss; abdominal pain; low blood pressure"),
    ):
        chunks.append({"chunk_id": cid, "document_id": doc_id, "question": question, "content": content,
                       "source_file": "test.xml", "source_url": "https://example.invalid/test"})
    return SimpleNamespace(chunks=chunks, chunks_sha256="test-sha", documents=[{}]*200)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.payload = mock_payload()
        self.candidates = cg.make_candidates(self.payload)

    def test_14_candidates_have_verified_evidence(self):
        self.assertEqual(len(self.candidates), 14)
        self.assertEqual(len({r["claim_key"] for r in self.candidates}), 14)
        self.assertTrue(all(r["chunk_content_sha256"] for r in self.candidates))

    def test_modified_evidence_detected(self):
        self.payload.chunks[0]["content"] = "text changed"
        with self.assertRaisesRegex(ValueError, "Phrase justificative"):
            cg.make_candidates(self.payload)

    def test_mismatched_document_detected(self):
        self.payload.chunks[2]["document_id"] = 99
        with self.assertRaisesRegex(ValueError, "Provenance"):
            cg.make_candidates(self.payload)

    def _write_review(self, path):
        path.write_text("\ufeff"+cg.make_review_csv(self.candidates), encoding="utf-8")
        return path

    def test_review_initially_pending(self):
        with TemporaryDirectory() as tmp:
            rows = cg.load_reviews(self._write_review(Path(tmp)/"review.csv"), self.candidates)
            self.assertEqual(sum(x["decision"]=="pending" for x in rows), 14)
            with self.assertRaisesRegex(ValueError, "Revue incomplete"):
                cg.approved_review(rows)

    def test_reject_mutated_evidence(self):
        with TemporaryDirectory() as tmp:
            file = self._write_review(Path(tmp)/"review.csv")
            text = file.read_text(encoding="utf-8-sig").replace("sleep apnea", "wrong fake symptom")
            file.write_text(text, encoding="utf-8-sig")
            with self.assertRaises(ValueError):
                cg.load_reviews(file, self.candidates)

    def test_approved_requires_reviewer(self):
        with TemporaryDirectory() as tmp:
            file = self._write_review(Path(tmp)/"review.csv")
            text = file.read_text(encoding="utf-8-sig").replace(",pending,,", ",approved,,", 1)
            file.write_text(text, encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "Reviewer requis"):
                cg.load_reviews(file, self.candidates)

    def test_mixed_approved_rejected(self):
        with TemporaryDirectory() as tmp:
            file = self._write_review(Path(tmp)/"review.csv")
            rows = list(csv.DictReader(io.StringIO(file.read_text(encoding="utf-8-sig"))))
            for i,row in enumerate(rows):
                row["decision"] = "approved" if i%2==0 else "rejected"
                row["reviewer"] = "test-reviewer"
            out=io.StringIO(newline="")
            w=csv.DictWriter(out, fieldnames=cg.REVIEW_FIELDS)
            w.writeheader();w.writerows(rows)
            file.write_text(out.getvalue(), encoding="utf-8-sig")
            validated = cg.load_reviews(file, self.candidates)
            approved=cg.approved_review(validated)
            self.assertEqual(len(approved), 7)

    def test_manifest_tamper_detected(self):
        with TemporaryDirectory() as tmp:
            f=Path(tmp)/"candidates.json"
            manifest=copy.deepcopy(cg.candidate_manifest(self.payload, self.candidates))
            manifest["candidates"][0]["symptom"]="invented"
            f.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "altere"):
                cg.read_manifest_file(f, self.payload, self.candidates)

    def test_transaction_rolls_back_if_evidence_missing(self):
        class FakeResult:
            def single(self):return {"matched_rows": 6}
        class FakeTx:
            def run(self, *a, **kw):return FakeResult()
        with self.assertRaisesRegex(RuntimeError, "annulee"):
            cg.execute_write_transaction(FakeTx(), self.candidates[:7], "sha")

    def test_write_cypher_no_deletes(self):
        code=cg.WRITE_CLAIMS.upper()
        self.assertNotIn("DELETE ", code)
        self.assertNotIn("DETACH", code)
        self.assertIn("SUPPORTED_BY", code)
        self.assertIn("CONTENT_SHA256", code)


if __name__ == "__main__":
    unittest.main()
