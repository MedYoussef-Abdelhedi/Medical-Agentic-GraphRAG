import unittest
import numpy as np
from generate_rag_eval_v3 import retrieve, make_context, verify_index, read_cases
from score_ragas_v3_async import validate_input, mean_or_none

class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        basis = np.zeros((3, 384), dtype=np.float32)
        basis[0, 0] = 1
        basis[1, 1] = 1
        basis[2, 2] = 1
        self.vectors = basis
        self.chunks = [{
            'chunk_id': f'{j}_0', 'document_id': j, 'question': 'Q',
            'content': f'Information {j}',
            'text_for_embedding': f'Question: Q\nAnswer: Information {j}',
            'source_file': f'{j}.xml', 'source_url': 'https://example.org'
        } for j in range(3)]

    def test_index_valid(self):
        verify_index(self.chunks, self.vectors)

    def test_index_mismatch_detected(self):
        with self.assertRaises(ValueError):
            verify_index(self.chunks[:-1], self.vectors)

    def test_retrieve_correct_order(self):
        query = np.zeros((384,), dtype=np.float32)
        query[2] = 1
        rows = retrieve('question', query, self.chunks, self.vectors)
        self.assertEqual(rows[0]['chunk_id'], '2_0')
        self.assertEqual(len(rows), 3)
        self.assertIn('[1]', make_context(rows))

    def test_invalid_eval_input(self):
        with self.assertRaises(ValueError):
            validate_input({'examples': [{'id': 'x', 'type': 'answerable', 'user_input': 'q',
                                          'response': 'r', 'retrieved_contexts': ['ctx']} ]})

    def test_mean(self):
        self.assertEqual(mean_or_none([0.5, 1.0]), .75)
        self.assertIsNone(mean_or_none([]))


if __name__ == '__main__':
    unittest.main()
