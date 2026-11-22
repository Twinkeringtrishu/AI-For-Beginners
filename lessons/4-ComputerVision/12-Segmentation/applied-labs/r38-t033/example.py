"""
R38-T033: Segmentation-mask overlap — compare two approaches
Lesson: lessons/4-ComputerVision/12-Segmentation (Segmentation)

Concept Connection:
In semantic segmentation (lessons/4-ComputerVision/12-Segmentation), each pixel in an image is
assigned a discrete integer class label. Rather than raw pixel accuracy (which can be heavily
biased by dominant background pixels), the standard evaluation metric is Intersection over
Union (IoU), also known as the Jaccard Index:

    IoU(c) = |Truth_c ∩ Pred_c| / |Truth_c ∪ Pred_c|
           = TP_c / (TP_c + FP_c + FN_c)

Absent-Class Policy:
When a class is present in neither the ground truth nor the predicted mask (|Truth_c ∪ Pred_c| == 0),
the IoU denominator is zero (0/0). This implementation defines an explicit policy:
- Policy: ABSENT_CLASS_POLICY = "sentinel"
- Behavior: Returns ABSENT_CLASS_SENTINEL = None
- Rationale: Classes with zero occurrences in both masks provide no spatial signal; returning
  None explicitly signals undefined overlap and excludes the absent class from mean-IoU (mIoU)
  calculations without causing ZeroDivisionError.

Two Approaches Compared:
1. Direct Boolean Mask Approach (calculate_iou_direct):
   Iterates through classes, creates element-wise boolean masks (truth == c) and (pred == c),
   and directly computes count(A & B) / count(A | B).
2. Independent Contingency/Confusion Matrix Approach (calculate_iou_contingency):
   Constructs a class-by-class contingency table counting True Positives (TP), False Positives (FP),
   and False Negatives (FN), computing TP / (TP + FP + FN).
Both approaches take identical inputs and produce mathematically identical results.

Scope and Simplifications:
This implementation operates on bounded 1D flattened integer masks using only the Python standard
library. It demonstrates evaluation arithmetic without requiring GPU inference or external
packages (such as PyTorch, OpenCV, or NumPy).
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Documented sentinel returned when a class is absent from both ground truth and prediction
ABSENT_CLASS_SENTINEL: None = None


def validate_masks(
    truth: Sequence[int],
    prediction: Sequence[int],
) -> None:
    """Validate that truth and prediction masks are non-empty sequences of equal length with valid integer labels."""
    if not isinstance(truth, (list, tuple)):
        raise TypeError(f"truth must be a list or tuple of integers, got {type(truth).__name__}")
    if not isinstance(prediction, (list, tuple)):
        raise TypeError(f"prediction must be a list or tuple of integers, got {type(prediction).__name__}")
    if len(truth) != len(prediction):
        raise ValueError(f"Mask length mismatch: truth has {len(truth)}, prediction has {len(prediction)}")
    if len(truth) == 0:
        raise ValueError("Masks cannot be empty")

    for idx, val in enumerate(truth):
        if isinstance(val, bool) or not isinstance(val, int):
            raise TypeError(f"truth[{idx}] must be an integer, got {type(val).__name__}")
        if val < 0:
            raise ValueError(f"truth[{idx}] cannot be negative, got {val}")

    for idx, val in enumerate(prediction):
        if isinstance(val, bool) or not isinstance(val, int):
            raise TypeError(f"prediction[{idx}] must be an integer, got {type(val).__name__}")
        if val < 0:
            raise ValueError(f"prediction[{idx}] cannot be negative, got {val}")


def get_target_classes(
    truth: Sequence[int],
    prediction: Sequence[int],
    classes: Optional[Sequence[int]] = None,
) -> List[int]:
    """Determine the list of classes to evaluate."""
    if classes is not None:
        if not isinstance(classes, (list, tuple, set)):
            raise TypeError("classes must be a sequence of integers")
        unique_classes = sorted(set(classes))
        for c in unique_classes:
            if isinstance(c, bool) or not isinstance(c, int):
                raise TypeError(f"class item must be integer, got {type(c).__name__}")
            if c < 0:
                raise ValueError(f"class item cannot be negative, got {c}")
        return unique_classes
    return sorted(set(truth).union(set(prediction)))


# ─────────────────────────────────────────────────────────────────────────────
# Approach 1: Direct Set-Theoretic / Boolean Masking
# ─────────────────────────────────────────────────────────────────────────────
def calculate_iou_direct(
    truth: Sequence[int],
    prediction: Sequence[int],
    classes: Optional[Sequence[int]] = None,
) -> Dict[int, Optional[float]]:
    """
    Compute per-class IoU directly using element-wise boolean mask intersections and unions.

    Returns a mapping: class_id -> IoU (or None if class is absent from both masks).
    """
    validate_masks(truth, prediction)
    target_classes = get_target_classes(truth, prediction, classes)
    results: Dict[int, Optional[float]] = {}

    for c in target_classes:
        intersection = 0
        union = 0
        for t, p in zip(truth, prediction):
            in_truth = (t == c)
            in_pred = (p == c)
            if in_truth and in_pred:
                intersection += 1
            if in_truth or in_pred:
                union += 1

        if union == 0:
            results[c] = ABSENT_CLASS_SENTINEL
        else:
            results[c] = intersection / union

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Approach 2: Independent Contingency / Confusion Table Lookup
# ─────────────────────────────────────────────────────────────────────────────
def calculate_iou_contingency(
    truth: Sequence[int],
    prediction: Sequence[int],
    classes: Optional[Sequence[int]] = None,
) -> Dict[int, Optional[float]]:
    """
    Compute per-class IoU independently using contingency statistics (TP, FP, FN).

    IoU = TP / (TP + FP + FN)
    Returns a mapping: class_id -> IoU (or None if class is absent from both masks).
    """
    validate_masks(truth, prediction)
    target_classes = get_target_classes(truth, prediction, classes)
    results: Dict[int, Optional[float]] = {}

    for c in target_classes:
        tp = 0
        fp = 0
        fn = 0
        for t, p in zip(truth, prediction):
            if t == c and p == c:
                tp += 1
            elif t != c and p == c:
                fp += 1
            elif t == c and p != c:
                fn += 1

        denominator = tp + fp + fn
        if denominator == 0:
            results[c] = ABSENT_CLASS_SENTINEL
        else:
            results[c] = tp / denominator

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Approach Comparison & Equivalence Verification
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ApproachComparisonResult:
    direct_iou: Dict[int, Optional[float]]
    contingency_iou: Dict[int, Optional[float]]
    is_equivalent: bool
    evaluated_classes: List[int]


def compare_approaches(
    truth: Sequence[int],
    prediction: Sequence[int],
    classes: Optional[Sequence[int]] = None,
) -> ApproachComparisonResult:
    """
    Run both approaches on identical inputs and verify mathematical equivalence.
    """
    res_direct = calculate_iou_direct(truth, prediction, classes)
    res_contingency = calculate_iou_contingency(truth, prediction, classes)

    target_classes = sorted(res_direct.keys())
    equivalent = True

    for c in target_classes:
        val1 = res_direct[c]
        val2 = res_contingency[c]
        if val1 is None or val2 is None:
            if val1 is not val2:
                equivalent = False
                break
        else:
            if not math.isclose(val1, val2, rel_tol=1e-9, abs_tol=1e-9):
                equivalent = False
                break

    return ApproachComparisonResult(
        direct_iou=res_direct,
        contingency_iou=res_contingency,
        is_equivalent=equivalent,
        evaluated_classes=target_classes,
    )


def compute_mean_iou(
    per_class_iou: Dict[int, Optional[float]],
    include_absent: bool = False,
) -> Optional[float]:
    """
    Compute mean Intersection-over-Union (mIoU) across classes.
    By default (include_absent=False), absent classes (None) are excluded from the mean.
    """
    valid_scores = [score for score in per_class_iou.values() if score is not None]
    if not valid_scores:
        return None
    return sum(valid_scores) / len(valid_scores)


# Canonical Stated Contract Fixture & Expected Values
CANONICAL_TRUTH = [0, 1, 1, 0]
CANONICAL_PREDICTION = [0, 1, 0, 0]
CANONICAL_CLASSES = [0, 1, 2]  # includes absent class 2 to test tie/absent rule

EXPECTED_CANONICAL_RESULTS = {
    0: 2 / 3,       # class 0: intersection=2, union=3 -> 2/3
    1: 1 / 2,       # class 1: intersection=1, union=2 -> 1/2
    2: None,        # class 2: absent from both -> None (sentinel)
}


def main() -> None:
    """Run demonstration of canonical scenario, absent-class policy, and approach comparison."""
    print("=" * 70)
    print("R38-T033: Segmentation-Mask Overlap — Compare Two Approaches")
    print("=" * 70)

    truth = CANONICAL_TRUTH
    pred = CANONICAL_PREDICTION
    evaluated_classes = CANONICAL_CLASSES

    print("\n--- Canonical Fixture ---")
    print(f"Ground Truth : {truth}")
    print(f"Prediction   : {pred}")
    print(f"Classes      : {evaluated_classes} (0: background, 1: object, 2: absent class)")

    # Approach 1: Direct boolean masking
    iou_direct = calculate_iou_direct(truth, pred, evaluated_classes)
    # Approach 2: Contingency table
    iou_contingency = calculate_iou_contingency(truth, pred, evaluated_classes)
    # Comparison
    comparison = compare_approaches(truth, pred, evaluated_classes)

    print("\n--- Approach 1: Direct Set/Mask Calculation ---")
    for c, score in iou_direct.items():
        score_repr = f"{score:.4f}" if score is not None else "None (absent sentinel)"
        print(f"  Class {c}: IoU = {score_repr}")

    print("\n--- Approach 2: Contingency (TP/(TP+FP+FN)) Calculation ---")
    for c, score in iou_contingency.items():
        score_repr = f"{score:.4f}" if score is not None else "None (absent sentinel)"
        print(f"  Class {c}: IoU = {score_repr}")

    print(f"\nMathematical Equivalence Verified: {comparison.is_equivalent}")
    print(f"Mean IoU (active classes 0 & 1)  : {compute_mean_iou(iou_direct):.4f}")


if __name__ == "__main__":
    main()
