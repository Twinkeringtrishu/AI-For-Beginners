# Applied Lab R38-T040: Token-Budget Preflight — Offline Service Adapter

**Lesson**: [5-NLP/17-GenerativeNetworks](../../README.md)  
**Task ID**: R38-T040

---

## Overview & Lesson Context

In Generative Networks and Large Language Models (LLMs), inference calls and prompt completion requests have strict context-length limits and resource budgets. Before sending prompts over an API network to a remote model service, applications perform **preflight token budgeting** to validate prompt length, manage token counts, and avoid service-side truncation or out-of-budget failures.

This lab places the external tokenization service behind a **callable adapter interface** (`TokenizerFn`) and provides a deterministic **offline local adapter** (`OfflineTokenBudgetAdapter`) for unit testing and local development without cloud dependencies.

---

## Architecture: Offline Service Adapter

The external tokenization dependency is abstracted behind a callable interface:
- **Interface**: `TokenizerFn = Callable[[str], List[str]]`
- **Adapter**: `OfflineTokenBudgetAdapter(tokenizer=..., adapter_name=...)`
- **Default Local Implementation**: `local_whitespace_tokenizer(text: str) -> List[str]`

This decoupling allows testing token-budget logic in an isolated, deterministic environment, with the ability to plug in real production tokenizers (e.g. TikToken, HuggingFace transformers) in production pipelines without changing calling code.

---

## Scope & Disclaimers

> [!WARNING]
> **Teaching Approximation Notice**:
> The local adapter uses basic whitespace splitting (`str.split()`) as an educational simulation.
> 
> **Real-service behaviors NOT established by this mock:**
> 1. **Subword / BPE Tokenization**: Real models (GPT, LLaMA) use Byte-Pair Encoding or WordPiece where words split into subword fragments.
> 2. **Context Window Enforcement**: Real cloud APIs strictly enforce combined prompt + completion token limits.
> 3. **API Rate Limiting & Auth**: No authentication headers, quota throttling, or network timeouts are simulated.
> 4. **Streaming & Partial Generation**: No streaming chunks or asynchronous response lifecycles.

---

## Stated Contract & Canonical Worked Example

Given a text prompt and an integer budget:
- **Canonical Input**: `"one two three four"`, `budget=3`
- **Total Tokens**: `4` (`['one', 'two', 'three', 'four']`)
- **Retained Tokens**: `['one', 'two', 'three']`
- **Removed Count**: `1` (`['four']`)
- **Is Truncated**: `True`
- **Empty Input**: `""`, `budget=3` retains `[]` with `0` removed.
- **Negative Budget**: explicitly rejected with `ValueError`.
- **Invalid Types**: rejected with `TypeError`.

---

## Verification & CLI Usage

```bash
# Run the worked example with execution evidence
python3 lessons/5-NLP/17-GenerativeNetworks/applied-labs/r38-t040/example.py

# Run the test suite
python3 -m unittest discover -s lessons/5-NLP/17-GenerativeNetworks/applied-labs/r38-t040 -p "test_*.py" -v
```

All 12 unit tests pass with zero external dependencies using Python standard library primitives only.
