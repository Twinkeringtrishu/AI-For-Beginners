"""
Unit tests for R38-T040: Token-budget preflight — offline service adapter.
Lesson: lessons/5-NLP/17-GenerativeNetworks
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from example import (
    TEACHING_DISCLAIMER,
    BudgetCheckResult,
    OfflineTokenBudgetAdapter,
    check_budget,
    get_worked_explanation,
    local_whitespace_tokenizer,
)


class TestOfflineTokenBudgetAdapter(unittest.TestCase):
    """Test suite for the offline token-budget adapter."""

    def setUp(self):
        self.adapter = OfflineTokenBudgetAdapter()

    def test_canonical_budget_3_retains_first_three(self):
        """Budget 3 with 'one two three four' retains [one,two,three], removes 1."""
        result = self.adapter.check("one two three four", 3)
        self.assertEqual(result.retained_tokens, ["one", "two", "three"])
        self.assertEqual(result.removed_tokens, ["four"])
        self.assertEqual(result.removed_count, 1)
        self.assertTrue(result.is_truncated)
        self.assertEqual(result.total_tokens, 4)

    def test_empty_input_retains_none(self):
        """Empty string: no tokens retained, nothing removed."""
        result = self.adapter.check("", 3)
        self.assertEqual(result.retained_tokens, [])
        self.assertEqual(result.removed_count, 0)
        self.assertFalse(result.is_truncated)

    def test_exact_budget_match_no_truncation(self):
        """When token count == budget, is_truncated is False."""
        result = self.adapter.check("alpha beta gamma", 3)
        self.assertEqual(result.retained_tokens, ["alpha", "beta", "gamma"])
        self.assertFalse(result.is_truncated)

    def test_under_budget_all_retained(self):
        """Fewer tokens than budget: all retained, nothing removed."""
        result = self.adapter.check("hello world", 5)
        self.assertEqual(result.retained_tokens, ["hello", "world"])
        self.assertEqual(result.removed_count, 0)

    def test_zero_budget_removes_all(self):
        """Zero budget retains nothing."""
        result = self.adapter.check("one two three", 0)
        self.assertEqual(result.retained_tokens, [])
        self.assertEqual(result.removed_count, 3)
        self.assertTrue(result.is_truncated)

    def test_negative_budget_raises_value_error(self):
        """Negative budget raises ValueError."""
        with self.assertRaises(ValueError):
            self.adapter.check("text", -1)

    def test_invalid_budget_type_raises_type_error(self):
        """Non-int budget (float, str, bool, None) raises TypeError."""
        for bad in (3.5, "3", True, False, None):
            with self.assertRaises(TypeError):
                self.adapter.check("text", bad)

    def test_invalid_text_type_raises_type_error(self):
        """Non-str text raises TypeError."""
        for bad in (123, None, ["a"], {"k": "v"}):
            with self.assertRaises(TypeError):
                self.adapter.check(bad, 3)

    def test_disclaimer_present_and_correct(self):
        """Every result carries the full teaching disclaimer."""
        result = self.adapter.check("sample text", 2)
        self.assertIn("Teaching approximation", result.disclaimer)
        self.assertIn("Not a real generative model tokenizer", result.disclaimer)
        self.assertEqual(result.disclaimer, TEACHING_DISCLAIMER)

    def test_adapter_name_exposed(self):
        """Adapter name is accessible on result."""
        result = self.adapter.check("hello", 5)
        self.assertEqual(result.adapter_name, "local_whitespace")

    def test_worked_explanation_regression(self):
        """Worked explanation contains canonical expected substrings."""
        expl = get_worked_explanation("one two three four", 3)
        self.assertIn("Token-Budget Preflight", expl)
        self.assertIn("'one two three four'", expl)
        self.assertIn("Budget: 3", expl)
        self.assertIn("['one', 'two', 'three']", expl)

    def test_deliberate_wrong_result_fails(self):
        """A wrong expected retained list must not match actual."""
        result = self.adapter.check("one two three four", 3)
        actual = result.retained_tokens
        wrong = ["one", "two"]
        self.assertNotEqual(actual, wrong)
        with self.assertRaises(AssertionError):
            self.assertEqual(actual, wrong)


if __name__ == "__main__":
    unittest.main()
