"""
R38-T032: Single perceptron update — boundary-case lesson extension
Lesson: lessons/3-NeuralNetworks/04-OwnFramework (Introduction to Neural Networks. Multi-Layered Perceptron)

Concept Connection:
In neural network architectures (lessons/3-NeuralNetworks/04-OwnFramework), the fundamental linear
layer computation is given by z = w * x + b. For binary classification with a single perceptron,
this linear pre-activation (score) is passed through a step threshold activation function:
    y_pred = 1 if score > 0 else 0 (with documented zero-score tie rule).

When a prediction error occurs (target != y_pred), the perceptron learning rule updates weights and bias:
    error = target - y_pred
    delta = learning_rate * error
    w_new = w + delta * x
    b_new = b + delta

Zero-Score Tie Rule & Boundary Decision:
When the weighted sum equals exactly zero (score == 0.0), a tie-breaking decision is required:
- Policy: score <= 0 -> 0; score > 0 -> 1.
- Exact boundary decision: At score == 0.0, the prediction is explicitly 0.
- Rationale: A zero score indicates zero net evidence for the positive class. When uninitialized
  weights (all zeros) and bias 0 encounter an input, the perceptron conservatively defaults to class 0,
  triggering an update if the target is 1.

Boundary Case Taxonomy:
1. Smallest Valid Fixture: 2D inputs x in R^2, binary target in {0, 1}, learning_rate > 0.
2. Boundary Cases:
   - Exact zero pre-activation score (score == 0.0 -> prediction = 0).
   - Near-zero pre-activations (+epsilon -> 1, -epsilon -> 0).
   - Zero error condition (target == prediction -> delta = 0, no weights/bias change).
   - Negative error update (target = 0, prediction = 1 -> weights decrease).
3. Invalid Cases:
   - Dimension mismatch (len(x) != 2).
   - Empty input vectors.
   - Non-binary targets (e.g. target=2, -1, 0.5).
   - Non-finite or non-numeric inputs (NaN, Inf, strings, booleans, None).
   - Non-positive learning rate (lr <= 0).

Scope and Simplifications:
This implementation is an isolated, minimal educational demonstration of a single forward-pass
and single-step update rule using only the Python standard library. It does not train a full
multilayer network, compute backpropagation gradients, or require GPUs/cloud dependencies.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Sequence, Tuple, Union


@dataclass
class PerceptronState:
    """Represents the parameters of a single perceptron."""
    weights: Tuple[float, float]
    bias: float
    learning_rate: float = 1.0


@dataclass
class PerceptronStepResult:
    """Encapsulates the complete state and arithmetic results of a single update step."""
    initial_score: float
    initial_prediction: int
    target: int
    error: int
    delta: float
    new_weights: Tuple[float, float]
    new_bias: float
    weights_updated: bool


def validate_input_vector(x: Sequence[Union[int, float]], expected_dim: int = 2, name: str = "input") -> None:
    """Validate that input is a sequence of finite real numbers matching expected dimension."""
    if not isinstance(x, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(x).__name__}")
    if len(x) != expected_dim:
        raise ValueError(f"{name} dimension mismatch: expected {expected_dim}, got {len(x)}")
    for idx, val in enumerate(x):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"{name}[{idx}] must be a numeric int/float, got {type(val).__name__}")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"{name}[{idx}] must be a finite real number, got {val}")


def validate_target(target: int) -> None:
    """Validate that target is binary (0 or 1)."""
    if isinstance(target, bool) or not isinstance(target, int):
        raise TypeError(f"target must be integer 0 or 1, got {type(target).__name__}")
    if target not in (0, 1):
        raise ValueError(f"target must be binary 0 or 1, got {target}")


def validate_learning_rate(lr: float) -> None:
    """Validate that learning rate is a positive finite number."""
    if isinstance(lr, bool) or not isinstance(lr, (int, float)):
        raise TypeError(f"learning_rate must be numeric, got {type(lr).__name__}")
    if math.isnan(lr) or math.isinf(lr) or lr <= 0:
        raise ValueError(f"learning_rate must be positive and finite, got {lr}")


def compute_score(weights: Sequence[Union[int, float]], bias: float, x: Sequence[Union[int, float]]) -> float:
    """
    Compute pre-activation linear score z = w0*x0 + w1*x1 + b.
    """
    validate_input_vector(x, expected_dim=len(weights), name="x")
    return sum(float(w) * float(val) for w, val in zip(weights, x)) + float(bias)


def predict(score: float) -> int:
    """
    Step activation function implementing the documented zero-score tie rule:
    - score > 0 -> 1
    - score <= 0 -> 0 (Exact zero score breaks ties toward 0)
    """
    if math.isnan(score) or math.isinf(score):
        raise ValueError(f"score must be finite, got {score}")
    return 1 if score > 0.0 else 0


def predict_score(state: PerceptronState, x: Sequence[Union[int, float]]) -> Tuple[float, int]:
    """Compute score and binary prediction for a given perceptron state and input."""
    score = compute_score(state.weights, state.bias, x)
    y_pred = predict(score)
    return score, y_pred


def perceptron_update_step(
    state: PerceptronState,
    x: Sequence[Union[int, float]],
    target: int,
) -> PerceptronStepResult:
    """
    Execute one single perceptron prediction and parameter update step.

    Stated Contract:
    - Initial weights: (0.0, 0.0)
    - Initial bias: 0.0
    - Input: [1, 2]
    - Target: 1
    - Zero-score tie rule produces initial prediction 0.
    - Learning rate 1.0 update gives new weights (1.0, 2.0) and bias 1.0.
    """
    validate_input_vector(x, expected_dim=len(state.weights), name="x")
    validate_target(target)
    validate_learning_rate(state.learning_rate)

    score, y_pred = predict_score(state, x)
    error = target - y_pred
    delta = float(state.learning_rate) * float(error)

    if error != 0:
        new_w0 = float(state.weights[0]) + delta * float(x[0])
        new_w1 = float(state.weights[1]) + delta * float(x[1])
        new_bias = float(state.bias) + delta
        updated = True
    else:
        new_w0, new_w1 = float(state.weights[0]), float(state.weights[1])
        new_bias = float(state.bias)
        updated = False

    return PerceptronStepResult(
        initial_score=score,
        initial_prediction=y_pred,
        target=target,
        error=error,
        delta=delta,
        new_weights=(new_w0, new_w1),
        new_bias=new_bias,
        weights_updated=updated,
    )


# Boundary Case Catalog for Documentation and Verification
BOUNDARY_CASE_CATALOG: List[Dict[str, Any]] = [
    {
        "category": "Smallest Valid (Canonical)",
        "description": "Initial zero weights with target=1: score=0 -> pred=0 -> update to weights [1,2], bias 1",
        "weights": (0.0, 0.0),
        "bias": 0.0,
        "input": [1, 2],
        "target": 1,
        "lr": 1.0,
        "expected_score": 0.0,
        "expected_pred": 0,
        "expected_weights": (1.0, 2.0),
        "expected_bias": 1.0,
    },
    {
        "category": "Boundary (Exact Zero Tie)",
        "description": "Zero score with target=0: score=0 -> pred=0 -> target=0, zero error, NO update",
        "weights": (0.0, 0.0),
        "bias": 0.0,
        "input": [1, 2],
        "target": 0,
        "lr": 1.0,
        "expected_score": 0.0,
        "expected_pred": 0,
        "expected_weights": (0.0, 0.0),
        "expected_bias": 0.0,
    },
    {
        "category": "Boundary (Near Zero Positive)",
        "description": "Score slightly positive (+1e-9): score > 0 -> pred=1",
        "weights": (1e-9, 0.0),
        "bias": 0.0,
        "input": [1.0, 0.0],
        "target": 1,
        "lr": 1.0,
        "expected_score": 1e-9,
        "expected_pred": 1,
        "expected_weights": (1e-9, 0.0),
        "expected_bias": 0.0,
    },
    {
        "category": "Boundary (Near Zero Negative)",
        "description": "Score slightly negative (-1e-9): score <= 0 -> pred=0",
        "weights": (-1e-9, 0.0),
        "bias": 0.0,
        "input": [1.0, 0.0],
        "target": 0,
        "lr": 1.0,
        "expected_score": -1e-9,
        "expected_pred": 0,
        "expected_weights": (-1e-9, 0.0),
        "expected_bias": 0.0,
    },
    {
        "category": "Boundary (Negative Update Direction)",
        "description": "Target=0, prediction=1: error=-1, weights decrease by lr*x",
        "weights": (1.0, 1.0),
        "bias": 0.0,
        "input": [1.0, 2.0],
        "target": 0,
        "lr": 0.5,
        "expected_score": 3.0,
        "expected_pred": 1,
        "expected_weights": (0.5, 0.0),
        "expected_bias": -0.5,
    },
]


def main() -> None:
    """Run demonstration of canonical scenario and boundary case catalog."""
    print("=" * 70)
    print("R38-T032: Single Perceptron Update Boundary-Case Lesson Extension")
    print("=" * 70)

    # 1. Mandatory Canonical Contract
    state = PerceptronState(weights=(0.0, 0.0), bias=0.0, learning_rate=1.0)
    x = [1, 2]
    target = 1

    print("\n--- 1. Canonical Stated Contract Execution ---")
    print(f"Initial State : weights={state.weights}, bias={state.bias}, learning_rate={state.learning_rate}")
    print(f"Input Sample  : x={x}, target={target}")

    result = perceptron_update_step(state, x, target)

    print(f"Forward Pass  : score = {result.initial_score:.4f}")
    print(f"Tie Rule Dec. : score <= 0 -> prediction = {result.initial_prediction}")
    print(f"Error & Delta : error = target - pred = {result.error}, delta = {result.delta}")
    print(f"Updated State : weights = {result.new_weights}, bias = {result.new_bias}")

    # 2. Boundary Cases Execution
    print("\n--- 2. Boundary Case Scenarios ---")
    for case in BOUNDARY_CASE_CATALOG:
        case_state = PerceptronState(weights=case["weights"], bias=case["bias"], learning_rate=case["lr"])
        case_res = perceptron_update_step(case_state, case["input"], case["target"])
        print(f"[{case['category']}] {case['description']}")
        print(f"  Score: {case_res.initial_score} -> Pred: {case_res.initial_prediction} (Target: {case_res.target})")
        print(f"  New Weights: {case_res.new_weights}, New Bias: {case_res.new_bias}")


if __name__ == "__main__":
    main()
