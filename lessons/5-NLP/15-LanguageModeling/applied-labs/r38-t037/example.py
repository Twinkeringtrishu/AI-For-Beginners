"""
R38-T037: Cosine-similarity lookup — failure and recovery example.
Lesson: lessons/5-NLP/15-LanguageModeling

Scope & Limitations:
This module demonstrates vector cosine similarity calculation, zero-vector
sentinel handling, and robust failure recovery for sequence/batch embedding lookups.
It uses only the Python standard library. It does not load large pre-trained word
embedding tables (e.g. 300D GloVe/Word2Vec) nor does it perform GPU-accelerated
linear algebra or neural network inference.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

UNDEFINED_SENTINEL: None = None


@dataclass(frozen=True)
class ComparisonOutcome:
    """Detailed record of a single vector comparison step."""
    step: int
    vec_a: Any
    vec_b: Any
    status: str  # "SUCCESS", "UNDEFINED", "ERROR"
    similarity: Optional[float]
    explanation: str
    recovered_from_previous_failure: bool


def vector_norm(vec: Sequence[float]) -> float:
    """Compute the Euclidean L2 norm of a vector."""
    if not isinstance(vec, (list, tuple)):
        raise TypeError(f"Vector must be a list or tuple, got {type(vec).__name__}")
    if len(vec) == 0:
        raise ValueError("Vector must not be empty")

    sum_sq = 0.0
    for idx, val in enumerate(vec):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"Vector elements must be numeric, got {type(val).__name__} at index {idx}")
        if not math.isfinite(val):
            raise ValueError(f"Vector elements must be finite, got {val} at index {idx}")
        sum_sq += float(val) ** 2

    return math.sqrt(sum_sq)


def dot_product(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute inner dot product between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}")

    total = 0.0
    for idx, (a, b) in enumerate(zip(vec_a, vec_b)):
        if isinstance(a, bool) or not isinstance(a, (int, float)):
            raise TypeError(f"vec_a element must be numeric, got {type(a).__name__} at index {idx}")
        if isinstance(b, bool) or not isinstance(b, (int, float)):
            raise TypeError(f"vec_b element must be numeric, got {type(b).__name__} at index {idx}")
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError(f"Vector elements must be finite at index {idx}: ({a}, {b})")
        total += float(a) * float(b)

    return total


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> Optional[float]:
    """
    Calculate the cosine similarity between two embedding vectors.

    Parameters
    ----------
    vec_a : Sequence[float]
        First vector.
    vec_b : Sequence[float]
        Second vector.

    Returns
    -------
    Optional[float]
        Cosine similarity in range [-1.0, 1.0], or UNDEFINED_SENTINEL (None)
        if either vector has zero Euclidean length (zero vector).

    Raises
    ------
    ValueError
        If vector dimensions do not match, vectors are empty, or contain non-finite numbers.
    TypeError
        If inputs are not lists/tuples of numeric types.
    """
    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)

    # If vector dimensions differ, dot_product will raise ValueError
    dot = dot_product(vec_a, vec_b)

    # Documented policy: zero vector returns UNDEFINED_SENTINEL (None)
    if norm_a == 0.0 or norm_b == 0.0:
        return UNDEFINED_SENTINEL

    denom = norm_a * norm_b
    sim = dot / denom

    # Clamp numerical precision artifacts
    if sim > 1.0:
        return 1.0
    if sim < -1.0:
        return -1.0
    return sim


def safe_compare(vec_a: Any, vec_b: Any, step: int, had_previous_failure: bool) -> ComparisonOutcome:
    """
    Execute a single vector comparison with error capture and recovery logging.
    """
    try:
        sim = cosine_similarity(vec_a, vec_b)
        if sim is UNDEFINED_SENTINEL:
            return ComparisonOutcome(
                step=step,
                vec_a=vec_a,
                vec_b=vec_b,
                status="UNDEFINED",
                similarity=None,
                explanation="Zero vector has norm 0.0; cosine similarity direction is undefined -> returned sentinel (None).",
                recovered_from_previous_failure=had_previous_failure,
            )
        return ComparisonOutcome(
            step=step,
            vec_a=vec_a,
            vec_b=vec_b,
            status="SUCCESS",
            similarity=sim,
            explanation=f"Calculated valid cosine similarity: {sim:.4f}.",
            recovered_from_previous_failure=had_previous_failure,
        )
    except (ValueError, TypeError) as err:
        return ComparisonOutcome(
            step=step,
            vec_a=vec_a,
            vec_b=vec_b,
            status="ERROR",
            similarity=None,
            explanation=f"Invalid/corrupted data: {type(err).__name__}: {err}",
            recovered_from_previous_failure=False,
        )


def run_pipeline(pairs: List[Tuple[Any, Any]]) -> List[ComparisonOutcome]:
    """
    Process a stream of vector pairs, recording outcomes and verifying recovery.
    """
    outcomes: List[ComparisonOutcome] = []
    previous_was_failure = False

    for idx, (va, vb) in enumerate(pairs, start=1):
        outcome = safe_compare(va, vb, step=idx, had_previous_failure=previous_was_failure)
        outcomes.append(outcome)
        previous_was_failure = outcome.status == "ERROR"

    return outcomes


def get_canonical_demo_pipeline() -> List[ComparisonOutcome]:
    """
    Construct the canonical sequence demonstrating:
    1. [1, 0] vs [1, 0] -> 1.0 (Identical)
    2. [1, 0] vs [0, 1] -> 0.0 (Orthogonal)
    3. [1, 0] vs [0, 0] -> None (Zero vector sentinel)
    4. [1, 0] vs [1, 0, 99] -> Dimension mismatch error (Documented invalid case)
    5. [0, 1] vs [0, 1] -> 1.0 (Successful recovery on next valid input)
    """
    test_stream: List[Tuple[Any, Any]] = [
        ([1, 0], [1, 0]),           # Step 1: Normal contract -> 1.0
        ([1, 0], [0, 1]),           # Step 2: Normal contract -> 0.0
        ([1, 0], [0, 0]),           # Step 3: Zero vector sentinel -> None
        ([1, 0], [1, 0, 99]),       # Step 4: Invalid dimension mismatch -> Error handled
        ([0, 1], [0, 1]),           # Step 5: Recovery -> 1.0
    ]
    return run_pipeline(test_stream)


def main() -> None:
    """Run demonstration pipeline and persist execution logs."""
    outcomes = get_canonical_demo_pipeline()

    print("==================================================================")
    print("R38-T037: Cosine Similarity Failure & Recovery Pipeline Demo")
    print("==================================================================")
    for out in outcomes:
        rec_tag = " [RECOVERED]" if out.recovered_from_previous_failure else ""
        print(f"Step {out.step} | Status: {out.status:<9} | Sim: {str(out.similarity):<6}{rec_tag}")
        print(f"  Inputs:      vec_a={out.vec_a}, vec_b={out.vec_b}")
        print(f"  Explanation: {out.explanation}\n")

    # Persist evidence to repo-root work/R38-T037/example_run.txt
    repo_root = Path(__file__).resolve().parents[5]
    work_dir = repo_root / "work" / "R38-T037"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_file = work_dir / "example_run.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("R38-T037 Cosine Similarity Pipeline Run Output\n\n")
        json.dump([asdict(o) for o in outcomes], f, indent=2)
        f.write("\n")

    print(f"Persisted evidence to: {out_file}")


if __name__ == "__main__":
    main()
