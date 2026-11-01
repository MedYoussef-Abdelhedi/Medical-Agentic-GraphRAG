"""Unit tests using standard library unittest; no downloads or real embeddings."""
import unittest
import numpy as np
from evaluate_retrieval import get_gold_ids, first_gold_rank, metric_summary, evaluate_version

class RetrievalBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"chunk_id":"5_0", "document_id":5,"content":"A common symptom is thirst"},
            {"chunk_id":"5_1", "document_id":5,"content":"Long-term monitoring is important"},
            {"chunk_id":"6_0", "document_id":6,"content":"The test measures hormone X"},
        ]
        self.vectors = np.array([[1.0,0.0],[0.0,1.0],[0.7,0.7]], dtype=np.float32)
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)

    def test_gold_evidence_matches_within_doc(self):
        example = {"reference_document_id":5, "reference_evidence_phrase":"  common   SYMPTOM is thirst"}
        self.assertEqual(get_gold_ids(example, self.chunks), {"5_0"})

    def test_first_gold_rank(self):
        order = np.array([1,2,0])
        self.assertEqual(first_gold_rank(order,self.chunks,{"5_0"}),3)
        self.assertIsNone(first_gold_rank(order[:2],self.chunks,{"5_0"}))

    def test_metrics(self):
        x = metric_summary([1,2,None,3])
        self.assertEqual(x['AnchorHit@1'],.25)
        self.assertEqual(x['AnchorHit@3'],.75)
        self.assertAlmostEqual(x['AnchorMRR@5'],(1+.5+1/3)/4)

    def test_unsupported_probe_excluded(self):
        from evaluate_retrieval import MAX_K
        examples = [
            {"id":"P", "type":"answerable", "domain":"test", "query":"question", "reference_document_id":5,"reference_evidence_phrase":"A common symptom is thirst"},
            {"id":"N", "type":"unsupported_probe", "domain":"unsupported", "query":"secret record"},
        ]
        q = np.array([[1.,0.],[0.,1.]], dtype=np.float32)
        _, rows = evaluate_version("SYNTH",self.chunks,self.vectors,examples,q)
        self.assertEqual(rows[0]["gold_rank_top5"], 1)
        self.assertIsNone(rows[1]["gold_rank_top5"])

    def test_fail_on_missing_anchor(self):
        example = [{"id":"P", "type":"answerable", "domain":"test", "query":"question", "reference_document_id":5,"reference_evidence_phrase":"phrase not in content"}]
        with self.assertRaisesRegex(ValueError, "evidence anchor not found"):
            evaluate_version("SYNTH",self.chunks,self.vectors,example,np.array([[1.,0.]],dtype=np.float32))

if __name__ == "__main__":
    unittest.main()
