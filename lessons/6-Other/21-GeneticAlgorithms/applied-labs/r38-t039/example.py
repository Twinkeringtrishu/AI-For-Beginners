"""
R38-T039: One genetic-selection round — scenario assessment.
Lesson: lessons/6-Other/21-GeneticAlgorithms

Disclaimer:
This is a deterministic teaching simulation of one fitness-ranking and
elite-selection round. It does NOT represent a full genetic algorithm,
real evolutionary computation, GPU-accelerated search, or trained model
performance. Tie-breaking is by candidate ID (lexicographic order).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TEACHING_DISCLAIMER: str = (
    "Teaching approximation: single-round deterministic fitness ranking and "
    "elite selection only. Not a full genetic algorithm or trained model."
)


@dataclass(frozen=True)
class Candidate:
    """An immutable candidate record with an ID and fitness score."""
    candidate_id: str
    fitness: float


@dataclass(frozen=True)
class SelectionResult:
    """Result of one elite-selection round."""
    population: List[Candidate]
    elite_count: int
    elites: List[Candidate]
    non_elites: List[Candidate]
    disclaimer: str = TEACHING_DISCLAIMER


def rank_by_fitness(population: List[Candidate]) -> List[Candidate]:
    """Return population sorted descending by fitness, ties broken by candidate_id.

    Args:
        population: List of Candidate records.

    Returns:
        New list sorted by (fitness DESC, candidate_id ASC).
    """
    if not isinstance(population, list):
        raise TypeError(f"population must be a list, got {type(population).__name__}")
    for c in population:
        if not isinstance(c, Candidate):
            raise TypeError(f"All items must be Candidate instances, got {type(c).__name__}")
    return sorted(population, key=lambda c: (-c.fitness, c.candidate_id))


def select_elites(population: List[Candidate], elite_count: int) -> SelectionResult:
    """Select the top elite_count candidates by fitness (ties broken by ID).

    Args:
        population:  List of Candidate records.
        elite_count: Number of elites to retain. Must be >= 0 and <= len(population).

    Returns:
        A frozen SelectionResult.

    Raises:
        TypeError:  If elite_count is not an int or is a bool.
        ValueError: If elite_count is negative or exceeds population size.
    """
    if not isinstance(population, list):
        raise TypeError(f"population must be a list, got {type(population).__name__}")
    if isinstance(elite_count, bool) or not isinstance(elite_count, int):
        raise TypeError(f"elite_count must be an int, got {type(elite_count).__name__}")
    if elite_count < 0:
        raise ValueError(f"elite_count must be >= 0, got {elite_count}")
    if elite_count > len(population):
        raise ValueError(
            f"elite_count ({elite_count}) exceeds population size ({len(population)})"
        )

    ranked = rank_by_fitness(population)
    elites = ranked[:elite_count]
    non_elites = ranked[elite_count:]

    return SelectionResult(
        population=list(population),
        elite_count=elite_count,
        elites=elites,
        non_elites=non_elites,
        disclaimer=TEACHING_DISCLAIMER,
    )


def get_scenario_assessment(
    population: List[Candidate], elite_count: int
) -> str:
    """Generate a scenario assessment report for one selection round."""
    result = select_elites(population, elite_count)
    ranked = rank_by_fitness(population)

    lines = [
        "=" * 68,
        "R38-T039: One Genetic-Selection Round — Scenario Assessment",
        "=" * 68,
        f"Population size: {len(population)}, Elite count: {elite_count}",
        "",
        "Q1 — Predict output: Who are the elites?",
        f"   Ranked: {[(c.candidate_id, c.fitness) for c in ranked]}",
        f"   Elites: {[(c.candidate_id, c.fitness) for c in result.elites]}",
        "",
        "Q2 — Diagnose failure: What if elite_count > population?",
        f"   elite_count={len(population)+1} raises ValueError — "
        "count beyond population is rejected.",
        "",
        "Q3 — Boundary policy: What if two candidates share equal fitness?",
        "   Ties are broken by candidate_id (lexicographic/ascending order),",
        "   ensuring deterministic, reproducible selection.",
        "",
        f"8. Note: {TEACHING_DISCLAIMER}",
        "=" * 68,
    ]
    return "\n".join(lines)


def main() -> None:
    """Run canonical contract, scenario assessment and write evidence."""
    # Canonical fixture: fitness A=3, B=8, C=5; elite_count=2 → selects B, C
    population = [
        Candidate("A", 3),
        Candidate("B", 8),
        Candidate("C", 5),
    ]
    elite_count = 2
    result = select_elites(population, elite_count)

    print(get_scenario_assessment(population, elite_count))
    print("\nStructured JSON Result:")
    print(json.dumps(asdict(result), indent=2))

    repo_root = Path(__file__).resolve().parents[5]
    work_dir = repo_root / "work" / "R38-T039"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_file = work_dir / "example_run.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(get_scenario_assessment(population, elite_count))
        f.write("\n\nStructured Result:\n")
        json.dump(asdict(result), f, indent=2)
        f.write("\n")

    print(f"\nWrote execution evidence to: {out_file}")


if __name__ == "__main__":
    main()
