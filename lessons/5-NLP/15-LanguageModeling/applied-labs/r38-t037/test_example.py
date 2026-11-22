"""
Unit tests for R38-T037: Cosine-similarity lookup — failure and recovery example.
Lesson: lessons/5-NLP/15-LanguageModeling
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from example import (
    UNDEFINED_SENTINEL,
    cosine_similarity,
    dot_product,
    get_canonical_demo_pipeline,
    run_pipeline,
    vector_norm,
)


class TestCosineSimilarityFailureAndRecovery(unittest.TestCase):
    """Test suite verifying cosine similarity contracts, zero-vector sentinel, and failure recovery."""

    def test_identical_vectors_yield_one(self):
        """[1,0] versus [1,0] gives 1."""
        res = cosine_similarity([1, 0], [1, 0])
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res, 1.0, places=7)

    def test_orthogonal_vectors_yield_zero(self):
        """[1,0] versus [0,1] gives 0."""
        res = cosine_similarity([1, 0], [0, 1])
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res, 0.0, places=7)

    def test_zero_vector_returns_undefined_sentinel(self):
        """A zero vector returns the documented undefined-result sentinel (None)."""
        # Right vector zero
        self.assertIs(cosine_similarity([1, 0], [0, 0]), UNDEFINED_SENTINEL)
        # Left vector zero
        self.assertIs(cosine_similarity([0, 0], [1, 0]), UNDEFINED_SENTINEL)
        # Both vectors zero
        self.assertIs(cosine_similarity([0, 0], [0, 0]), UNDEFINED_SENTINEL)
        # Multi-dimensional zero
        self.assertIs(cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]), UNDEFINED_SENTINEL)

    def test_opposite_vectors_yield_negative_one(self):
        """Opposite direction vectors yield -1.0."""
        res = cosine_similarity([1, 0], [-1, 0])
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res, -1.0, places=7)

    def test_arbitrary_vector_similarity(self):
        """Known 2D vector cosine similarity: [3, 4] vs [4, 3] = 24 / 25 = 0.96."""
        res = cosine_similarity([3, 4], [4, 3])
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res, 0.96, places=7)

    def test_dimension_mismatch_raises_value_error(self):
        """Vectors of differing lengths raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            cosine_similarity([1, 0], [1, 0, 0])
        self.assertIn("dimension mismatch", str(cm.exception).lower())

    def test_empty_vectors_raise_value_error(self):
        """Empty vectors raise ValueError."""
        with self.assertRaises(ValueError):
            cosine_similarity([], [])

    def test_non_finite_values_raise_value_error(self):
        """NaN or infinite values raise ValueError."""
        with self.assertRaises(ValueError):
            cosine_similarity([float("nan"), 1.0], [1.0, 0.0])
        with self.assertRaises(ValueError):
            cosine_similarity([float("inf"), 1.0], [1.0, 0.0])

    def test_non_numeric_elements_raise_type_error(self):
        """Booleans, strings, or None values raise TypeError."""
        with self.assertRaises(TypeError):
            cosine_similarity([True, False], [1, 0])  # type: ignore
        with self.assertRaises(TypeError):
            cosine_similarity(["a", "b"], [1, 0])  # type: ignore
        with self.assertRaises(TypeError):
            cosine_similarity([None, 1], [1, 0])  # type: ignore

    def test_failure_and_recovery_demonstrated(self):
        """
        Focus: Exercise a documented invalid/missing-data case, explain the error,
        and demonstrate recovery on the next valid input.
        """
        outcomes = get_canonical_demo_pipeline()
        self.assertEqual(len(outcomes), 5)

        # Step 1: Normal contract -> 1.0
        self.assertEqual(outcomes[0].status, "SUCCESS")
        self.assertAlmostEqual(outcomes[0].similarity, 1.0, places=7)

        # Step 2: Normal contract -> 0.0
        self.assertEqual(outcomes[1].status, "SUCCESS")
        self.assertAlmostEqual(outcomes[1].similarity, 0.0, places=7)

        # Step 3: Zero vector sentinel -> None
        self.assertEqual(outcomes[2].status, "UNDEFINED")
        self.assertIs(outcomes[2].similarity, UNDEFINED_SENTINEL)

        # Step 4: Invalid/missing data error handled and explained
        self.assertEqual(outcomes[3].status, "ERROR")
        self.assertIsNone(outcomes[3].similarity)
        self.assertIn("ValueError", outcomes[3].explanation)
        self.assertIn("dimension mismatch", outcomes[3].explanation.lower())
        self.assertFalse(outcomes[3].recovered_from_previous_failure)

        # Step 5: Successful recovery on next valid input
        self.assertEqual(outcomes[4].status, "SUCCESS")
        self.assertAlmostEqual(outcomes[4].similarity, 1.0, places=7)
        self.assertTrue(outcomes[4].recovered_from_previous_failure)

    def test_custom_pipeline_recovery_from_corrupt_type(self):
        """Pipeline catches TypeError on corrupt input and recovers on the next valid step."""
        pairs = [
            ([1.0, 0.0], [1.0, 0.0]),
            ("corrupt-string", [1.0, 0.0]),  # type error
            ([0.0, 1.0], [0.0, 1.0]),        # recovery
        ]
        outcomes = run_pipeline(pairs)
        self.assertEqual(outcomes[0].status, "SUCCESS")
        self.assertEqual(outcomes[1].status, "ERROR")
        self.assertIn("TypeError", outcomes[1].explanation)
        self.assertEqual(outcomes[2].status, "SUCCESS")
        self.assertTrue(outcomes[2].recovered_from_previous_failure)

    def test_deliberate_wrong_result_fails(self):
        """
        Verify that a deliberate wrong result makes the test fail.
        Done When criterion: 'a deliberate wrong result makes the test fail'.
        """
        res = cosine_similarity([1, 0], [1, 0])
        actual_sim = res
        wrong_sim = 0.5  # Actual is 1.0

        self.assertNotEqual(actual_sim, wrong_sim)
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_sim, wrong_sim)


if __name__ == "__main__":
    unittest.main()
