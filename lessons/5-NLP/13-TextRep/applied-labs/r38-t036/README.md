# Applied Lab R38-T036: Token-Budget Preflight — Worked Example with Checks

**Lesson**: [5-NLP/13-TextRep](../../README.md)  
**Task ID**: R38-T036

---

## Overview & Lesson Context

In Natural Language Processing (NLP), before raw text can be converted to tensors, vectors, or input sequences, it must first be tokenized into discrete units (characters, subwords, or words) and mapped against model capacity limits. Real-world neural architectures and language models impose strict maximum context lengths or input token budgets (e.g. 512, 2048, or 8192 tokens).

This lab introduces **preflight token-budget validation**—evaluating prompt length against a configured token budget *before* costly downstream tensor allocation or model inference begins. It provides clear, deterministic reporting of retained tokens, truncated tokens, and boundary validations.

---

## Teaching Simplifications & Disclaimers

> [!WARNING]
> **Teaching Approximation Notice**:
> The tokenizer used here is a lightweight, deterministic whitespace-splitting rule (`str.split()`) designed strictly for educational clarity. It is **not** a production subword tokenizer (such as BPE, WordPiece, or SentencePiece) and does **not** simulate real hardware acceleration, cloud service latency, or model inference parameters.

In production:
- Subword tokenizers split arbitrary words into morphemes and byte fragments.
- Tokenizers inject special tokens (e.g., `[CLS]`, `[SEP]`, `<|endoftext|>`).
- Truncation policies can truncate from the left, right, or center, or require sliding window chunks.

---

## Stated Contract & Worked Example

Given a text string and an integer `budget >= 0`:
1. The text is tokenized into whitespace-delimited tokens.
2. If `len(tokens) <= budget`, all tokens are retained with `removed_count = 0`.
3. If `len(tokens) > budget`, the first `budget` tokens are retained and the remainder are reported as truncated (`removed_count = len(tokens) - budget`).
4. An empty string (`""`) or whitespace-only string produces zero tokens and zero removed tokens.
5. A negative budget (`budget < 0`) is explicitly rejected with a `ValueError`.
6. Non-integer budgets or non-string inputs are rejected with a `TypeError`.

### Canonical Worked Example
- **Input Text**: `"one two three four"`
- **Configured Budget**: `3`
- **Total Tokens**: `4` (`["one", "two", "three", "four"]`)
- **Retained Tokens**: `["one", "two", "three"]`
- **Removed Count**: `1` (`["four"]`)
- **Is Truncated**: `True`

---

## File Structure

- `example.py`: Core tokenization, budget enforcement, structured `TokenBudgetResult`, and worked explanation.
- `test_example.py`: Full unit test suite covering canonical behavior, boundary conditions, type validation, regression check, and deliberate wrong-result rejection.

---

## CLI Usage & Verification

### Running the Example
```bash
python3 lessons/5-NLP/13-TextRep/applied-labs/r38-t036/example.py
```

### Running Unit Tests
```bash
python3 -m unittest discover -s lessons/5-NLP/13-TextRep/applied-labs/r38-t036 -p "test_*.py" -v
```

All unit tests run deterministically with zero external dependencies and standard library tools only.
