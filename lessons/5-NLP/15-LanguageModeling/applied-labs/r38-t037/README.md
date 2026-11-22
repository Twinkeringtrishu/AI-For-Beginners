# Applied Lab R38-T037: Cosine-Similarity Lookup — Failure and Recovery Example

**Lesson**: [5-NLP/15-LanguageModeling](../../README.md)  
**Task ID**: R38-T037

---

## Overview & Lesson Context

In Language Modeling, word embeddings (such as CBoW, Skip-gram, or GloVe) project discrete vocabulary tokens into continuous vector spaces where semantic similarity corresponds to geometric proximity. The primary similarity metric between semantic vector embeddings is **cosine similarity**, defined as:

$$\text{cosine\_similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

When training embeddings or serving similarity lookups in real-world pipelines:
1. **Zero Norm / Zero Vectors**: Out-of-vocabulary tokens or uninitialized weights may yield all-zero vectors. Because $\|\mathbf{0}\|_2 = 0$, the denominator evaluates to zero. Dividing by zero is mathematically undefined. An explicit, documented policy must handle this by returning a sentinel value (`None`).
2. **Corrupted or Missing Data**: In continuous text streams, malformed vectors, dimension mismatches, or missing entries can occur. Pipelines must safely isolate errors, explain them clearly, and seamlessly recover on subsequent valid inputs without crashing.

---

## Scope & Simplifications

> [!NOTE]
> This applied lab uses pure Python standard library primitives to illustrate vector normalization, dot product calculation, sentinel policies for zero vectors, and pipeline error recovery. It does not download multi-gigabyte pre-trained models (e.g. BERT/GPT/Word2Vec), does not require GPUs or cloud providers, and operates on transparent local vector fixtures.

---

## Stated Contract

- **Identical non-zero vectors**: `[1, 0]` vs `[1, 0]` $\rightarrow$ `1.0`.
- **Orthogonal non-zero vectors**: `[1, 0]` vs `[0, 1]` $\rightarrow$ `0.0`.
- **Zero vectors**: If either vector has norm `0.0` (e.g. `[0, 0]`), cosine similarity returns `None` (`UNDEFINED_SENTINEL`).
- **Invalid data**: Dimension mismatches or non-numeric types raise explicit errors (`ValueError` / `TypeError`).
- **Pipeline Recovery**: Step-by-step stream processing handles errors and verifies recovery on the subsequent valid input.

---

## Canonical Pipeline Execution

| Step | Vector A | Vector B | Status | Cosine Sim | Notes |
|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `[1, 0]` | `[1, 0]` | `SUCCESS` | `1.0` | Identical vectors |
| 2 | `[1, 0]` | `[0, 1]` | `SUCCESS` | `0.0` | Orthogonal vectors |
| 3 | `[1, 0]` | `[0, 0]` | `UNDEFINED` | `None` | Zero-norm sentinel |
| 4 | `[1, 0]` | `[1, 0, 99]` | `ERROR` | `None` | Dimension mismatch (2 vs 3) caught & explained |
| 5 | `[0, 1]` | `[0, 1]` | `SUCCESS` | `1.0` | **Recovered** on subsequent valid input |

---

## Invocation & Verification

### Running the Example
```bash
python3 lessons/5-NLP/15-LanguageModeling/applied-labs/r38-t037/example.py
```

### Running the Test Suite
```bash
python3 -m unittest discover -s lessons/5-NLP/15-LanguageModeling/applied-labs/r38-t037 -p "test_*.py" -v
```
All 12 unit tests pass deterministically with zero external dependencies.
