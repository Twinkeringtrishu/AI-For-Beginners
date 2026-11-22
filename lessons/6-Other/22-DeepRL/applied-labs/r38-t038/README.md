# Applied Lab R38-T038: Finite-State Reward Trace — Evaluation Experiment

**Lesson**: [6-Other/22-DeepRL](../../README.md)  
**Task ID**: R38-T038

---

## Overview & Lesson Context

In Reinforcement Learning (RL), an agent interacts with an environment modeled as a Markov Decision Process (MDP). Unlike supervised learning, feedback is typically sparse and delayed: rewards are obtained only when specific state transitions occur (such as reaching a goal or terminal state). Moreover, practical RL training requires **finite episode horizons** (step caps) to bound trajectories and avoid infinite exploration loops.

This applied lab implements a 3-state chain MDP with:
1. Discrete actions (`left`, `right`).
2. Sparse terminal reward: $+1$ reward provided strictly upon entering terminal state `2`.
3. Bounded episode horizon: hard step cap at `5` steps.
4. Evaluation experiment comparing three controlled policy configurations with sample efficiency metrics.

---

## Scope & Simplifications

> [!NOTE]
> This lab provides a clear, deterministic Python standard library simulation for analyzing state transitions, reward traces, and horizon truncation. It does not use OpenAI Gym, deep neural networks (PyTorch/TensorFlow), GPU acceleration, or continuous physics simulators.

---

## Environment Specification

- **State Space**: $\mathcal{S} = \{0, 1, 2\}$, with $2$ designated as terminal.
- **Action Space**: $\mathcal{A} = \{\text{'left'}, \text{'right'}\}$.
- **Transitions**:
  - From $0$: `'left'` $\rightarrow 0$ (bounded); `'right'` $\rightarrow 1$ (reward 0).
  - From $1$: `'left'` $\rightarrow 0$ (reward 0); `'right'` $\rightarrow 2$ (reward 1, terminal).
  - From $2$: Episode terminates immediately.
- **Horizon Cap**: Default $\text{max\_steps} = 5$.

---

## Controlled Configurations Comparison

| Configuration | Action Sequence | Steps | Total Reward | Goal Reached? | Sample Efficiency ($\frac{\text{Reward}}{\text{Steps}}$) | Trade-off Analysis |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Config-A (Optimal Greedy)** | `["right", "right"]` | 2 | 1 | Yes | **0.500** | Direct path, maximum efficiency, zero wasted computation. |
| **Config-B (Hesitant / Exploration)** | `["right", "left", "right", "right"]` | 4 | 1 | Yes | **0.250** | Recovers from backtrack; reaches goal with halved sample efficiency. |
| **Config-C (Adversarial / Left)** | `["left"] * 5` | 5 | 0 | No | **0.000** | Fails to reach goal; step cap prevents infinite looping. |

---

## Invocation & Verification

```bash
# Run the evaluation experiment
python3 lessons/6-Other/22-DeepRL/applied-labs/r38-t038/example.py

# Run unit tests
python3 -m unittest discover -s lessons/6-Other/22-DeepRL/applied-labs/r38-t038 -p "test_*.py" -v
```
All 12 unit tests pass deterministically.
