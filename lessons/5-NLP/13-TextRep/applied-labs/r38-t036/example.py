"""
R38-T036: Token-budget preflight — worked example with checks.
Lesson: lessons/5-NLP/13-TextRep

Disclaimer:
This tokenizer is a deterministic teaching approximation using whitespace splitting.
It is NOT a production or real model tokenizer (such as Byte-Pair Encoding (BPE),
WordPiece, or SentencePiece) and does not represent real hardware, cloud service
behavior, or trained-model inference limits.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

TEACHING_DISCLAIMER: str = (
    "Teaching approximation: local whitespace tokenizer only. "
    "Not a real model tokenizer (BPE/WordPiece/SentencePiece)."
)


@dataclass(frozen=True)
class TokenBudgetResult:
    """Structured report of preflight token-budget validation."""
    input_text: str
    budget: int
    total_tokens: int
    retained_tokens: List[str]
    removed_tokens: List[str]
    removed_count: int
    is_truncated: bool
    disclaimer: str = TEACHING_DISCLAIMER


def tokenize(text: str) -> List[str]:
    """
    Deterministically split input text into whitespace-delimited tokens.

    Parameters
    ----------
    text : str
        Input string to tokenize.

    Returns
    -------
    List[str]
        List of whitespace tokens. Empty string returns an empty list.

    Raises
    ------
    TypeError
        If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Input text must be a str, got {type(text).__name__}")
    return text.split()


def apply_budget(tokens: List[str], budget: int) -> Tuple[List[str], int]:
    """
    Check tokens against budget and return retained tokens and truncation count.

    Parameters
    ----------
    tokens : List[str]
        List of string tokens.
    budget : int
        Maximum allowable token count. Must be non-negative.

    Returns
    -------
    retained : List[str]
        Tokens kept within the budget.
    removed_count : int
        Number of tokens truncated/removed.

    Raises
    ------
    TypeError
        If budget is not an integer (or is a boolean).
    ValueError
        If budget is negative.
    """
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError(f"Budget must be an int, got {type(budget).__name__}")
    if budget < 0:
        raise ValueError(f"Budget must be non-negative, got {budget}")

    if budget >= len(tokens):
        return list(tokens), 0

    retained = list(tokens[:budget])
    removed_count = len(tokens) - budget
    return retained, removed_count


def check_budget(text: str, budget: int) -> TokenBudgetResult:
    """
    Preflight check for prompt length against a configured token budget.

    Parameters
    ----------
    text : str
        Input text prompt.
    budget : int
        Configured token budget.

    Returns
    -------
    TokenBudgetResult
        Full report containing retained tokens, removed tokens, and status flags.
    """
    tokens = tokenize(text)
    retained, removed_count = apply_budget(tokens, budget)
    removed_tokens = list(tokens[budget:]) if removed_count > 0 else []

    return TokenBudgetResult(
        input_text=text,
        budget=budget,
        total_tokens=len(tokens),
        retained_tokens=retained,
        removed_tokens=removed_tokens,
        removed_count=removed_count,
        is_truncated=removed_count > 0,
        disclaimer=TEACHING_DISCLAIMER,
    )


def process(text: str, budget: int) -> Tuple[List[str], int]:
    """
    Convenience wrapper returning (retained_tokens, removed_count).
    """
    res = check_budget(text, budget)
    return res.retained_tokens, res.removed_count


def get_worked_explanation(text: str = "one two three four", budget: int = 3) -> str:
    """
    Return a step-by-step worked explanation from input string to output.
    """
    tokens = tokenize(text)
    res = check_budget(text, budget)
    explanation = [
        "============================================================",
        "Worked Input-to-Output Explanation (Token-Budget Preflight)",
        "============================================================",
        f"1. Input String: {text!r}",
        f"2. Configured Budget: {budget}",
        f"3. Tokenization Step: Whitespace split produces {len(tokens)} tokens: {tokens}",
        f"4. Length Comparison: Total tokens ({len(tokens)}) > Budget ({budget}) -> Truncation needed.",
        f"5. Slicing Step: Retain first {budget} tokens: tokens[:{budget}] = {res.retained_tokens}",
        f"6. Truncation Report: Discarded {res.removed_count} token(s): {res.removed_tokens}",
        f"7. Output: Retained = {res.retained_tokens}, Removed Count = {res.removed_count}",
        f"8. Note: {res.disclaimer}",
        "============================================================",
    ]
    return "\n".join(explanation)


def main() -> None:
    """Run canonical demonstration and write evidence file."""
    text = "one two three four"
    budget = 3
    result = check_budget(text, budget)

    print(get_worked_explanation(text, budget))
    print("\nStructured JSON Result:")
    print(json.dumps(asdict(result), indent=2))

    # Persist evidence to repo-root work/R38-T036/example_run.txt
    repo_root = Path(__file__).resolve().parents[5]
    work_dir = repo_root / "work" / "R38-T036"
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
