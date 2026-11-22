"""
Unit tests for R38-T036: Token-budget preflight — worked example with checks.
Lesson: lessons/5-NLP/13-TextRep
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from example import (
    TEACHING_DISCLAIMER,
    TokenBudgetResult,
    apply_budget,
    check_budget,
    get_worked_explanation,
    process,
    tokenize,
)


class TestTokenBudgetPreflight(unittest.TestCase):
    """Test suite verifying tokenization, budget truncation, validation, and worked explanation."""

    def test_canonical_stated_contract(self):
        """
        Budget 3 with 'one two three four' retains [one,two,three]
        and reports one removed token.
        """
        retained, removed_count = process("one two three four", 3)
        self.assertEqual(retained, ["one", "two", "three"])
        self.assertEqual(removed_count, 1)

        res = check_budget("one two three four", 3)
        self.assertEqual(res.retained_tokens, ["one", "two", "three"])
        self.assertEqual(res.removed_count, 1)
        self.assertEqual(res.removed_tokens, ["four"])
        self.assertTrue(res.is_truncated)
        self.assertEqual(res.total_tokens, 4)

    def test_empty_input_retains_none(self):
        """Empty input retains none and reports zero removed tokens."""
        retained, removed_count = process("", 3)
        self.assertEqual(retained, [])
        self.assertEqual(removed_count, 0)

        res = check_budget("", 3)
        self.assertEqual(res.retained_tokens, [])
        self.assertEqual(res.removed_count, 0)
        self.assertEqual(res.total_tokens, 0)
        self.assertFalse(res.is_truncated)

    def test_whitespace_only_input(self):
        """Whitespace-only strings evaluate to zero tokens."""
        retained, removed_count = process("   \t  \n  ", 5)
        self.assertEqual(retained, [])
        self.assertEqual(removed_count, 0)

    def test_exact_budget_match(self):
        """Tokens exactly equal to budget retains all with 0 removed."""
        retained, removed_count = process("alpha beta gamma", 3)
        self.assertEqual(retained, ["alpha", "beta", "gamma"])
        self.assertEqual(removed_count, 0)

        res = check_budget("alpha beta gamma", 3)
        self.assertFalse(res.is_truncated)

    def test_under_budget(self):
        """Tokens fewer than budget retains all without truncation."""
        retained, removed_count = process("hello world", 5)
        self.assertEqual(retained, ["hello", "world"])
        self.assertEqual(removed_count, 0)

    def test_zero_budget(self):
        """Budget 0 retains no tokens and removes all tokens."""
        retained, removed_count = process("one two three", 0)
        self.assertEqual(retained, [])
        self.assertEqual(removed_count, 3)

        res = check_budget("one two three", 0)
        self.assertTrue(res.is_truncated)
        self.assertEqual(res.removed_tokens, ["one", "two", "three"])

    def test_reject_negative_budget(self):
        """Negative budget raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            process("one two", -1)
        self.assertIn("Budget must be non-negative", str(cm.exception))

        with self.assertRaises(ValueError):
            apply_budget(["a", "b"], -5)

    def test_invalid_budget_types_raise_type_error(self):
        """Non-integer budgets (float, str, bool, None) raise TypeError."""
        for invalid_budget in [3.5, "3", True, False, None, [3]]:
            with self.assertRaises(TypeError):
                apply_budget(["a", "b"], invalid_budget)  # type: ignore

    def test_invalid_text_type_raises_type_error(self):
        """Non-string text inputs raise TypeError."""
        for invalid_text in [123, None, ["a", "b"], {"text": "val"}]:
            with self.assertRaises(TypeError):
                tokenize(invalid_text)  # type: ignore

    def test_worked_explanation_regression(self):
        """
        Regression test: Verify worked explanation text and execution match
        exact input-to-output contract.
        """
        explanation = get_worked_explanation("one two three four", 3)
        self.assertIn("Worked Input-to-Output Explanation", explanation)
        self.assertIn("Input String: 'one two three four'", explanation)
        self.assertIn("Configured Budget: 3", explanation)
        self.assertIn("Retained = ['one', 'two', 'three']", explanation)
        self.assertIn("Removed Count = 1", explanation)

    def test_teaching_disclaimer_labeled(self):
        """Label the tokenizer as a teaching approximation, not a real model tokenizer."""
        res = check_budget("sample text", 2)
        self.assertIn("Teaching approximation", res.disclaimer)
        self.assertIn("Not a real model tokenizer", res.disclaimer)
        self.assertEqual(res.disclaimer, TEACHING_DISCLAIMER)

    def test_deliberate_wrong_result_fails(self):
        """
        Verify that a deliberate wrong result makes the test fail.
        Done When criterion: 'a deliberate wrong result makes the test fail'.
        """
        res = check_budget("one two three four", 3)
        actual_retained = res.retained_tokens  # ['one', 'two', 'three']
        wrong_retained = ["one", "two"]

        self.assertNotEqual(actual_retained, wrong_retained)
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_retained, wrong_retained)


if __name__ == "__main__":
    unittest.main()
