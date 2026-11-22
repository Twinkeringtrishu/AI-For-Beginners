"""
R38-T040: Token-budget preflight — offline service adapter.
Lesson: lessons/5-NLP/17-GenerativeNetworks

Disclaimer:
This tokenizer is a deterministic teaching approximation using whitespace splitting.
It is NOT a production or real model tokenizer (BPE/WordPiece/SentencePiece),
does not represent real hardware, cloud service behavior, or trained-model inference.
The adapter interface mocks a generative-network token-budget service locally.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

TEACHING_DISCLAIMER: str = (
    "Teaching approximation: local whitespace tokenizer adapter only. "
    "Not a real generative model tokenizer or cloud service. "
    "Real-service behaviors NOT established by this mock: "
    "BPE/subword tokenization, context window enforcement, "
    "API rate limits, authentication, and streaming output."
)

# ---------------------------------------------------------------------------
# Adapter interface and local implementation
# ---------------------------------------------------------------------------

TokenizerFn = Callable[[str], List[str]]


def local_whitespace_tokenizer(text: str) -> List[str]:
    """Deterministic local adapter: splits on whitespace only.

    This is the offline mock. It does NOT replicate real model tokenization.

    Args:
        text: Input string.

    Returns:
        List of whitespace-delimited tokens.

    Raises:
        TypeError: If text is not a str.
    """
    if not isinstance(text, str):
        raise TypeError(f"Input text must be a str, got {type(text).__name__}")
    return text.split()


@dataclass(frozen=True)
class BudgetCheckResult:
    """Structured result of an offline token-budget preflight check."""
    input_text: str
    budget: int
    total_tokens: int
    retained_tokens: List[str]
    removed_tokens: List[str]
    removed_count: int
    is_truncated: bool
    adapter_name: str
    disclaimer: str = TEACHING_DISCLAIMER


class OfflineTokenBudgetAdapter:
    """Callable adapter wrapping any tokenizer behind a budget-check interface.

    This decouples the budget logic from the tokenizer implementation,
    allowing the real tokenizer to be swapped in without changing callers.

    Real-service behaviors this mock does NOT establish:
    - BPE/subword tokenization
    - Context window enforcement at the model API layer
    - API rate limits and authentication
    - Streaming output or partial results
    """

    def __init__(
        self,
        tokenizer: TokenizerFn = local_whitespace_tokenizer,
        adapter_name: str = "local_whitespace",
    ) -> None:
        self._tokenizer = tokenizer
        self.adapter_name = adapter_name

    def check(self, text: str, budget: int) -> BudgetCheckResult:
        """Run preflight token-budget check.

        Args:
            text:   Raw input string.
            budget: Maximum token count. Must be a non-negative int.

        Returns:
            BudgetCheckResult with retained/removed tokens and truncation info.

        Raises:
            TypeError:  If text is not str, or budget is not int (bool rejected).
            ValueError: If budget is negative.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError(f"budget must be an int, got {type(budget).__name__}")
        if budget < 0:
            raise ValueError(f"budget must be non-negative, got {budget}")

        tokens = self._tokenizer(text)

        if budget >= len(tokens):
            retained = list(tokens)
            removed: List[str] = []
            removed_count = 0
        else:
            retained = list(tokens[:budget])
            removed = list(tokens[budget:])
            removed_count = len(tokens) - budget

        return BudgetCheckResult(
            input_text=text,
            budget=budget,
            total_tokens=len(tokens),
            retained_tokens=retained,
            removed_tokens=removed,
            removed_count=removed_count,
            is_truncated=removed_count > 0,
            adapter_name=self.adapter_name,
            disclaimer=TEACHING_DISCLAIMER,
        )


# Module-level convenience using the default local adapter
_default_adapter = OfflineTokenBudgetAdapter()


def check_budget(text: str, budget: int) -> BudgetCheckResult:
    """Module-level shortcut using the default offline adapter."""
    return _default_adapter.check(text, budget)


def get_worked_explanation(text: str = "one two three four", budget: int = 3) -> str:
    """Return a human-readable walkthrough of the offline budget check."""
    result = check_budget(text, budget)
    lines = [
        "=" * 68,
        "R38-T040: Token-Budget Preflight — Offline Service Adapter",
        "=" * 68,
        f"Adapter: {result.adapter_name}",
        f"1. Input: {text!r}",
        f"2. Budget: {budget}",
        f"3. Tokenization (whitespace split): {result.total_tokens} tokens → {_default_adapter._tokenizer(text)}",
        f"4. Retained ({len(result.retained_tokens)}): {result.retained_tokens}",
        f"5. Removed  ({result.removed_count}): {result.removed_tokens}",
        f"6. Truncated: {result.is_truncated}",
        f"7. Note: {result.disclaimer}",
        "=" * 68,
    ]
    return "\n".join(lines)


def main() -> None:
    """Run canonical verification and write evidence."""
    adapter = OfflineTokenBudgetAdapter()

    text = "one two three four"
    budget = 3
    result = adapter.check(text, budget)

    print(get_worked_explanation(text, budget))
    print("\nStructured JSON Result:")
    print(json.dumps(asdict(result), indent=2))

    repo_root = Path(__file__).resolve().parents[5]
    work_dir = repo_root / "work" / "R38-T040"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_file = work_dir / "example_run.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(get_worked_explanation(text, budget))
        f.write("\n\nStructured Result:\n")
        json.dump(asdict(result), f, indent=2)
        f.write("\n")

    print(f"\nWrote execution evidence to: {out_file}")


if __name__ == "__main__":
    main()
