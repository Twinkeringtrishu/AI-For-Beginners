# Execution Evidence: R38-T036

**Task**: R38-T036 (6 of 10)  
**Lesson**: `lessons/5-NLP/13-TextRep`  
**Description**: Token-budget preflight — worked example with checks  

---

## 1. Environment & Tooling Versions

- **Python**: `Python 3.15.2`
- **Git**: `git version 2.55.0`
- **Dependencies**: None (Python standard library only)

---

## 2. Example Run Output

```text
============================================================
Worked Input-to-Output Explanation (Token-Budget Preflight)
============================================================
1. Input String: 'one two three four'
2. Configured Budget: 3
3. Tokenization Step: Whitespace split produces 4 tokens: ['one', 'two', 'three', 'four']
4. Length Comparison: Total tokens (4) > Budget (3) -> Truncation needed.
5. Slicing Step: Retain first 3 tokens: tokens[:3] = ['one', 'two', 'three']
6. Truncation Report: Discarded 1 token(s): ['four']
7. Output: Retained = ['one', 'two', 'three'], Removed Count = 1
8. Note: Teaching approximation: local whitespace tokenizer only. Not a real model tokenizer (BPE/WordPiece/SentencePiece).
============================================================

Structured JSON Result:
{
  "input_text": "one two three four",
  "budget": 3,
  "total_tokens": 4,
  "retained_tokens": [
    "one",
    "two",
    "three"
  ],
  "removed_tokens": [
    "four"
  ],
  "removed_count": 1,
  "is_truncated": true,
  "disclaimer": "Teaching approximation: local whitespace tokenizer only. Not a real model tokenizer (BPE/WordPiece/SentencePiece)."
}
```

---

## 3. Unit Test Execution Output

```text
test_canonical_stated_contract (test_example.TestTokenBudgetPreflight.test_canonical_stated_contract)
Budget 3 with 'one two three four' retains [one,two,three] ... ok
test_deliberate_wrong_result_fails (test_example.TestTokenBudgetPreflight.test_deliberate_wrong_result_fails)
Verify that a deliberate wrong result makes the test fail. ... ok
test_empty_input_retains_none (test_example.TestTokenBudgetPreflight.test_empty_input_retains_none)
Empty input retains none and reports zero removed tokens. ... ok
test_exact_budget_match (test_example.TestTokenBudgetPreflight.test_exact_budget_match)
Tokens exactly equal to budget retains all with 0 removed. ... ok
test_invalid_budget_types_raise_type_error (test_example.TestTokenBudgetPreflight.test_invalid_budget_types_raise_type_error)
Non-integer budgets (float, str, bool, None) raise TypeError. ... ok
test_invalid_text_type_raises_type_error (test_example.TestTokenBudgetPreflight.test_invalid_text_type_raises_type_error)
Non-string text inputs raise TypeError. ... ok
test_reject_negative_budget (test_example.TestTokenBudgetPreflight.test_reject_negative_budget)
Negative budget raises ValueError. ... ok
test_teaching_disclaimer_labeled (test_example.TestTokenBudgetPreflight.test_teaching_disclaimer_labeled)
Label the tokenizer as a teaching approximation, not a real model tokenizer. ... ok
test_under_budget (test_example.TestTokenBudgetPreflight.test_under_budget)
Tokens fewer than budget retains all without truncation. ... ok
test_whitespace_only_input (test_example.TestTokenBudgetPreflight.test_whitespace_only_input)
Whitespace-only strings evaluate to zero tokens. ... ok
test_worked_explanation_regression (test_example.TestTokenBudgetPreflight.test_worked_explanation_regression)
Regression test: Verify worked explanation text and execution match ... ok
test_zero_budget (test_example.TestTokenBudgetPreflight.test_zero_budget)
Budget 0 retains no tokens and removes all tokens. ... ok

----------------------------------------------------------------------
Ran 12 tests in 0.001s

OK
```

---

## 4. Verification of Criteria

- [x] Budget 3 with "one two three four" retains `[one,two,three]` and reports one removed token.
- [x] Empty input retains none (`[]`, 0 removed).
- [x] Reject negative budget (`ValueError`).
- [x] Label tokenizer as teaching approximation, not model's real tokenizer.
- [x] Lesson link and example agree on inputs, outputs, and prerequisite tools.
- [x] Contribution focus demonstrated (worked explanation + regression test).
- [x] Deliberate wrong result makes test fail (`AssertionError` verified).
- [x] No unsupported claims about real hardware, cloud services, or trained models.
