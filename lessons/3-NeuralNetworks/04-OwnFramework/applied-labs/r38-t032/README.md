# Applied Lab R38-T032: Single Perceptron Update — Boundary-Case Lesson Extension

## Overview & Lesson Concept Connection

In [Lesson 4: Introduction to Neural Networks. Multi-Layered Perceptron](../../README.md), the fundamental linear layer operation computes the weighted sum:

$$z = \mathbf{w} \cdot \mathbf{x} + b = \sum_{i=0}^{n-1} w_i x_i + b$$

Before extending to multilayer networks with non-linear activation functions and gradient backpropagation, this lab studies the elementary building block: **a single perceptron prediction and weight update step** on a 2-dimensional feature fixture with explicit boundary and tie-breaking rules.

---

## Zero-Score Tie Rule & Boundary Decision

The classical perceptron activation function maps the linear pre-activation score $z$ to a binary prediction $\hat{y} \in \{0, 1\}$.

When the score is exactly zero ($z = 0.0$), a documented tie-breaking decision is required:
- **Decision Policy**: $\hat{y} = 1 \iff z > 0$; otherwise $\hat{y} = 0 \iff z \le 0$.
- **Exact Boundary Decision**: At $z = 0.0$, the perceptron predicts $\hat{y} = 0$.
- **Rationale**: A score of zero signifies zero net evidence toward the positive class. When uninitialized parameters ($\mathbf{w} = [0, 0]$, $b = 0$) receive an input $\mathbf{x} = [1, 2]$, $z = 0 \cdot 1 + 0 \cdot 2 + 0 = 0.0$. Defaulting to $\hat{y} = 0$ ensures the perceptron detects a misclassification when the true target is $y = 1$, driving the learning rule to adjust the parameters.

---

## Perceptron Learning Rule

When prediction $\hat{y}$ does not equal true target $y \in \{0, 1\}$:
1. **Error**: $e = y - \hat{y} \in \{-1, 0, 1\}$
2. **Delta**: $\Delta = \eta \cdot e$ (where $\eta$ is the learning rate, $\eta > 0$)
3. **Weight Update**: $w_i \leftarrow w_i + \Delta \cdot x_i$
4. **Bias Update**: $b \leftarrow b + \Delta$

If $\hat{y} = y$, $e = 0 \implies \Delta = 0$, so no parameter updates occur.

---

## Boundary Case Taxonomy

| Category | Fixture / Condition | Initial Score & Prediction | Expected Result / Behavior |
| :--- | :--- | :--- | :--- |
| **Smallest Valid (Canonical)** | $\mathbf{w}=[0, 0], b=0, \mathbf{x}=[1, 2], y=1, \eta=1$ | $z = 0.0 \to \hat{y} = 0$ | $e=1, \Delta=1.0 \implies \mathbf{w}=[1.0, 2.0], b=1.0$ |
| **Boundary (Exact Zero Tie)** | $\mathbf{w}=[0, 0], b=0, \mathbf{x}=[1, 2], y=0, \eta=1$ | $z = 0.0 \to \hat{y} = 0$ | $e=0 \implies$ No update ($\mathbf{w}=[0, 0], b=0$) |
| **Boundary (Near-Zero Positive)** | $\mathbf{w}=[10^{-9}, 0], b=0, \mathbf{x}=[1, 0], y=1$ | $z = 10^{-9} > 0 \to \hat{y} = 1$ | $e=0 \implies$ No update |
| **Boundary (Near-Zero Negative)** | $\mathbf{w}=[-10^{-9}, 0], b=0, \mathbf{x}=[1, 0], y=0$ | $z = -10^{-9} \le 0 \to \hat{y} = 0$ | $e=0 \implies$ No update |
| **Boundary (Negative Update)** | $\mathbf{w}=[1, 1], b=1, \mathbf{x}=[1, 1], y=0, \eta=0.5$ | $z = 3.0 \to \hat{y} = 1$ | $e=-1, \Delta=-0.5 \implies \mathbf{w}=[0.5, 0.5], b=0.5$ |
| **Invalid Dimension** | $\mathbf{x} = [1]$ or $\mathbf{x} = [1, 2, 3]$ | N/A | Raises `ValueError` |
| **Invalid Target** | $y \notin \{0, 1\}$ (e.g. $y = 2$ or $y = -1$) | N/A | Raises `ValueError` |
| **Invalid Type** | $\mathbf{x} = [1, \text{"two"}]$ or $y = 0.5$ | N/A | Raises `TypeError` |
| **Non-finite Values** | $\mathbf{x} = [\text{NaN}, 2]$ or $\mathbf{x} = [\text{Inf}, 2]$ | N/A | Raises `ValueError` |
| **Invalid Learning Rate** | $\eta \le 0$ or $\eta = \text{Inf}$ | N/A | Raises `ValueError` |

---

## Scope & Limitations

- **Standard Library Only**: Implemented with pure Python standard library modules (`math`, `dataclasses`, `typing`); no external packages required.
- **Single-Step Focus**: Emphasizes one deterministic update step on local fixtures rather than training a multi-epoch classifier over an entire dataset.
- **Hardware Agnostic**: Fully self-contained local CPU execution with zero network, cloud, or GPU requirements.

---

## Execution Instructions

Run the concept example script:
```bash
python3 lessons/3-NeuralNetworks/04-OwnFramework/applied-labs/r38-t032/example.py
```

Run the automated test suite:
```bash
python3 -m unittest discover -s lessons/3-NeuralNetworks/04-OwnFramework/applied-labs/r38-t032 -p "test_*.py" -v
```
