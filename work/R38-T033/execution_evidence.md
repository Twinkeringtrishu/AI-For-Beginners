# Execution Evidence for Issue #33 (R38-T033)

- **Student**: Tanishka .
- **Task**: R38-T033: segmentation-mask overlap — compare two approaches [lessons/4-ComputerVision/12-Segmentation]
- **Repository**: https://github.com/S-AI-Projects/AI-For-Beginners
- **Branch**: `R38-T033`
- **Lab Directory**: `lessons/4-ComputerVision/12-Segmentation/applied-labs/r38-t033`
- **Environment**: Python 3.15.2, git version 2.55.0

---

## 1. Automated Test Suite Execution

```bash
python3 -m unittest discover -s lessons/4-ComputerVision/12-Segmentation/applied-labs/r38-t033 -p "test_*.py" -v
```

### Output:
```
test_absent_class_returns_sentinel (test_example.TestSegmentationMaskOverlap.test_absent_class_returns_sentinel)
Explicit policy for classes absent from both truth and prediction: ... ok
test_both_approaches_equivalent_canonical (test_example.TestSegmentationMaskOverlap.test_both_approaches_equivalent_canonical)
Verify that direct and contingency approaches produce identical outputs on canonical fixture. ... ok
test_both_approaches_equivalent_multiclass (test_example.TestSegmentationMaskOverlap.test_both_approaches_equivalent_multiclass)
Verify equivalence on complex multi-class fixture with 4 classes and partial overlaps. ... ok
test_canonical_stated_contract (test_example.TestSegmentationMaskOverlap.test_canonical_stated_contract)
Stated scenario: ... ok
test_compute_mean_iou (test_example.TestSegmentationMaskOverlap.test_compute_mean_iou)
compute_mean_iou properly averages present classes and ignores absent classes. ... ok
test_deliberate_wrong_result_fails (test_example.TestSegmentationMaskOverlap.test_deliberate_wrong_result_fails)
Verify that a deliberate wrong result makes the test fail. ... ok
test_dimension_mismatch_raises_value_error (test_example.TestSegmentationMaskOverlap.test_dimension_mismatch_raises_value_error)
Masks of unequal length raise ValueError. ... ok
test_empty_masks_raise_value_error (test_example.TestSegmentationMaskOverlap.test_empty_masks_raise_value_error)
Empty masks raise ValueError. ... ok
test_negative_labels_raise_value_error (test_example.TestSegmentationMaskOverlap.test_negative_labels_raise_value_error)
Negative class IDs raise ValueError. ... ok
test_non_integer_elements_raise_type_error (test_example.TestSegmentationMaskOverlap.test_non_integer_elements_raise_type_error)
Floats, booleans, or strings in masks raise TypeError. ... ok
test_perfect_overlap (test_example.TestSegmentationMaskOverlap.test_perfect_overlap)
Identical masks yield 1.0 IoU for all present classes. ... ok
test_zero_overlap (test_example.TestSegmentationMaskOverlap.test_zero_overlap)
Completely non-overlapping masks yield 0.0 IoU. ... ok

----------------------------------------------------------------------
Ran 12 tests in 0.000s

OK
```

---

## 2. Example Execution Output

```bash
python3 lessons/4-ComputerVision/12-Segmentation/applied-labs/r38-t033/example.py
```

### Output:
```
======================================================================
R38-T033: Segmentation-Mask Overlap — Compare Two Approaches
======================================================================

--- Canonical Fixture ---
Ground Truth : [0, 1, 1, 0]
Prediction   : [0, 1, 0, 0]
Classes      : [0, 1, 2] (0: background, 1: object, 2: absent class)

--- Approach 1: Direct Set/Mask Calculation ---
  Class 0: IoU = 0.6667
  Class 1: IoU = 0.5000
  Class 2: IoU = None (absent sentinel)

--- Approach 2: Contingency (TP/(TP+FP+FN)) Calculation ---
  Class 0: IoU = 0.6667
  Class 1: IoU = 0.5000
  Class 2: IoU = None (absent sentinel)

Mathematical Equivalence Verified: True
Mean IoU (active classes 0 & 1)  : 0.5833
```

---

## 3. Deliberate Failure Verification

Running a test case with an incorrect expectation exits with code 1:
```
AssertionError: 0.5 != 1.0
FAILED (failures=1)
Exit code: 1
```
