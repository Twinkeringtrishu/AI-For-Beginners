# Applied Lab R38-T039: One Genetic-Selection Round — Scenario Assessment

**Lesson**: [6-Other/21-GeneticAlgorithms](../../README.md)  
**Task ID**: R38-T039

---

## Overview & Lesson Context

Genetic Algorithms (GAs) rely on selection mechanisms to identify high-performing candidates from a population based on evaluated fitness scores. One core selection technique is **elite selection**, where the top $k$ candidates are preserved without alteration to guarantee that optimal genetic material is not lost across generations.

This lab implements a deterministic, immutable candidate ranking and elite selection round in pure Python, coupled with a **scenario assessment** addressing three applied engineering questions.

---

## Stated Contract & Deterministic Rules

1. **Candidate Immutability**: Each candidate is represented by an immutable record (`Candidate(candidate_id, fitness)`).
2. **Fitness Ranking**: Candidates are ranked descending by fitness.
3. **Deterministic Tie-Breaking**: When candidates share identical fitness scores, ties are deterministically resolved by sorting `candidate_id` in ascending (lexicographic) order.
4. **Boundary Validation**:
   - `elite_count < 0` $\rightarrow$ rejected with `ValueError`.
   - `elite_count > len(population)` $\rightarrow$ rejected with `ValueError`.
   - Non-integer or boolean types $\rightarrow$ rejected with `TypeError`.
5. **Canonical Fixture**:
   - Population: `A` (fitness 3), `B` (fitness 8), `C` (fitness 5).
   - `elite_count = 2` $\rightarrow$ selects `B` then `C`; `A` is placed in non-elites.

---

## Scenario Assessment: Three Applied Questions

| Question Type | Scenario | Answer & Rationale | Executable Check |
|:---|:---|:---|:---|
| **1. Predict Output** | Population $[(A,3), (B,8), (C,5)]$ with `elite_count=2`. | Elite candidates are `[B, C]` because fitness order is $8 > 5 > 3$. | `test_canonical_elite_selection_b_c` |
| **2. Diagnose Failure** | Configuration requests `elite_count=4` for 3 candidates. | Fails with `ValueError`: requesting more elites than available candidates violates population conservation. | `test_elite_count_exceeds_population_raises` |
| **3. Choose Boundary Policy** | Candidates $Z$ and $A$ have equal fitness $5.0$. | Deterministic policy: tie broken by lexicographical order of `candidate_id` ($A$ chosen before $Z$), ensuring reproducible selection. | `test_tie_broken_by_candidate_id` |

---

## Scope & Disclaimers

> [!NOTE]
> **Teaching Approximation Notice**:
> This example simulates a single deterministic fitness evaluation and elite selection step. It does **not** include multi-generation evolution, crossover, mutation, fitness landscapes, or GPU acceleration.

---

## Invocation & Verification

```bash
# Run the worked scenario assessment
python3 lessons/6-Other/21-GeneticAlgorithms/applied-labs/r38-t039/example.py

# Run unit tests
python3 -m unittest discover -s lessons/6-Other/21-GeneticAlgorithms/applied-labs/r38-t039 -p "test_*.py" -v
```

All 12 unit tests pass deterministically using standard library Python.
