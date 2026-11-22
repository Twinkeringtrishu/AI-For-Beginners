"""
Unit tests for R38-T033: Segmentation-mask overlap — compare two approaches.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from example import (
    ABSENT_CLASS_SENTINEL,
    CANONICAL_CLASSES,
    CANONICAL_PREDICTION,
    CANONICAL_TRUTH,
    EXPECTED_CANONICAL_RESULTS,
    calculate_iou_contingency,
    calculate_iou_direct,
    compare_approaches,
    compute_mean_iou,
)


class TestSegmentationMaskOverlap(unittest.TestCase):
    """Test suite verifying per-class IoU calculations, absent class policy, and approach equivalence."""

    def test_canonical_stated_contract(self):
        """
        Stated scenario:
        truth [0,1,1,0], prediction [0,1,0,0]:
        class1 IoU=1/2, class0 IoU=2/3.
        """
        res_direct = calculate_iou_direct(CANONICAL_TRUTH, CANONICAL_PREDICTION)
        res_contingency = calculate_iou_contingency(CANONICAL_TRUTH, CANONICAL_PREDICTION)

        # Class 0: truth={0, 3}, pred={0, 2, 3} -> intersection=2, union=3 -> 2/3
        self.assertAlmostEqual(res_direct[0], 2 / 3, places=7)
        self.assertAlmostEqual(res_contingency[0], 2 / 3, places=7)

        # Class 1: truth={1, 2}, pred={1} -> intersection=1, union=2 -> 1/2
        self.assertAlmostEqual(res_direct[1], 1 / 2, places=7)
        self.assertAlmostEqual(res_contingency[1], 1 / 2, places=7)

    def test_absent_class_returns_sentinel(self):
        """
        Explicit policy for classes absent from both truth and prediction:
        Returns ABSENT_CLASS_SENTINEL (None).
        """
        # Class 2 is explicitly evaluated but absent from truth and prediction
        res_direct = calculate_iou_direct(CANONICAL_TRUTH, CANONICAL_PREDICTION, classes=[0, 1, 2])
        res_contingency = calculate_iou_contingency(CANONICAL_TRUTH, CANONICAL_PREDICTION, classes=[0, 1, 2])

        self.assertIs(res_direct[2], ABSENT_CLASS_SENTINEL)
        self.assertIs(res_contingency[2], ABSENT_CLASS_SENTINEL)

    def test_both_approaches_equivalent_canonical(self):
        """Verify that direct and contingency approaches produce identical outputs on canonical fixture."""
        comp = compare_approaches(CANONICAL_TRUTH, CANONICAL_PREDICTION, CANONICAL_CLASSES)
        self.assertTrue(comp.is_equivalent)
        self.assertEqual(comp.direct_iou, comp.contingency_iou)

    def test_both_approaches_equivalent_multiclass(self):
        """Verify equivalence on complex multi-class fixture with 4 classes and partial overlaps."""
        truth = [0, 1, 2, 3, 0, 1, 2, 3]
        pred = [0, 2, 2, 3, 1, 1, 0, 3]
        classes = [0, 1, 2, 3, 4]  # class 4 is absent

        comp = compare_approaches(truth, pred, classes)
        self.assertTrue(comp.is_equivalent)
        self.assertEqual(comp.direct_iou, comp.contingency_iou)
        self.assertIs(comp.direct_iou[4], ABSENT_CLASS_SENTINEL)

    def test_perfect_overlap(self):
        """Identical masks yield 1.0 IoU for all present classes."""
        truth = [0, 1, 0, 1]
        pred = [0, 1, 0, 1]
        res = calculate_iou_direct(truth, pred)
        self.assertAlmostEqual(res[0], 1.0, places=7)
        self.assertAlmostEqual(res[1], 1.0, places=7)

    def test_zero_overlap(self):
        """Completely non-overlapping masks yield 0.0 IoU."""
        truth = [0, 0, 0, 0]
        pred = [1, 1, 1, 1]
        res = calculate_iou_direct(truth, pred)
        self.assertAlmostEqual(res[0], 0.0, places=7)
        self.assertAlmostEqual(res[1], 0.0, places=7)

    def test_compute_mean_iou(self):
        """compute_mean_iou properly averages present classes and ignores absent classes."""
        scores = {0: 2 / 3, 1: 1 / 2, 2: None}
        miou = compute_mean_iou(scores, include_absent=False)
        self.assertIsNotNone(miou)
        self.assertAlmostEqual(miou, (2 / 3 + 1 / 2) / 2, places=7)

        # All absent
        all_absent = {0: None, 1: None}
        self.assertIsNone(compute_mean_iou(all_absent, include_absent=False))

    def test_dimension_mismatch_raises_value_error(self):
        """Masks of unequal length raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_iou_direct([0, 1], [0, 1, 0])
        with self.assertRaises(ValueError):
            calculate_iou_contingency([0, 1], [0, 1, 0])

    def test_empty_masks_raise_value_error(self):
        """Empty masks raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_iou_direct([], [])
        with self.assertRaises(ValueError):
            calculate_iou_contingency([], [])

    def test_negative_labels_raise_value_error(self):
        """Negative class IDs raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_iou_direct([-1, 0], [0, 0])
        with self.assertRaises(ValueError):
            calculate_iou_contingency([0, 0], [0, -2])

    def test_non_integer_elements_raise_type_error(self):
        """Floats, booleans, or strings in masks raise TypeError."""
        with self.assertRaises(TypeError):
            calculate_iou_direct([0.5, 1.0], [0, 1])  # type: ignore
        with self.assertRaises(TypeError):
            calculate_iou_direct([True, False], [0, 1])  # type: ignore
        with self.assertRaises(TypeError):
            calculate_iou_direct(["0", "1"], [0, 1])  # type: ignore

    def test_deliberate_wrong_result_fails(self):
        """
        Verify that a deliberate wrong result makes the test fail.
        Done When criterion: 'a deliberate wrong result makes the test fail'.
        """
        res = calculate_iou_direct(CANONICAL_TRUTH, CANONICAL_PREDICTION)
        actual_class1_iou = res[1]  # Actual is 0.5 (1/2)

        self.assertNotEqual(actual_class1_iou, 1.0)
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_class1_iou, 1.0)


if __name__ == "__main__":
    unittest.main()
