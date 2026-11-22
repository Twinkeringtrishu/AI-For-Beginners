# Applied Lab R38-T033: Segmentation-Mask Overlap — Compare Two Approaches

## Overview & Lesson Concept Connection

In [Lesson 12: Segmentation](../../README.md), computer vision models advance from coarse bounding boxes to dense pixel-level classification (semantic segmentation).

In semantic segmentation:
- Every pixel in an image is assigned a discrete integer class label.
- Evaluating segmentation quality using standard accuracy is misleading because background pixels frequently dominate the image.
- Instead, the gold standard metric is **Intersection over Union (IoU)**, also called the **Jaccard Index**, evaluated per class:

$$\text{IoU}(c) = \frac{|\text{Truth}_c \cap \text{Pred}_c|}{|\text{Truth}_c \cup \text{Pred}_c|} = \frac{TP_c}{TP_c + FP_c + FN_c}$$

---

## Absent-Class Policy

When a specific class $c$ is present in **neither** the ground-truth mask **nor** the predicted mask:
- $|\text{Truth}_c \cup \text{Pred}_c| = 0$
- The fraction becomes $\frac{0}{0}$ (mathematically undefined).

### Documented Policy:
- Under `ABSENT_CLASS_POLICY = "sentinel"`, the function explicitly returns:
  ```python
  ABSENT_CLASS_SENTINEL = None
  ```
- **Rationale**: An absent class has zero spatial ground-truth instances and zero model false alarms. Returning `None` avoids arbitrary zero-division errors and ensures absent classes can be cleanly filtered out from mean-IoU ($\text{mIoU}$) computations across images.

---

## Two Approaches Compared

This lab implements and compares two distinct mathematical and algorithmic approaches to compute per-class IoU on identical inputs:

1. **Approach 1: Direct Set-Theoretic / Boolean Masking (`calculate_iou_direct`)**
   - For each class $c$, construct element-wise boolean masks: $\mathbf{m}_{\text{true}} = (\text{truth} == c)$ and $\mathbf{m}_{\text{pred}} = (\text{prediction} == c)$.
   - Direct counts: $\text{Intersection} = \sum (\mathbf{m}_{\text{true}} \land \mathbf{m}_{\text{pred}})$ and $\text{Union} = \sum (\mathbf{m}_{\text{true}} \lor \mathbf{m}_{\text{pred}})$.
   - $\text{IoU}(c) = \frac{\text{Intersection}}{\text{Union}}$.

2. **Approach 2: Contingency / Confusion Matrix Statistics (`calculate_iou_contingency`)**
   - For each class $c$, calculate confusion matrix counts across pixel positions:
     - True Positives ($TP_c$): $\text{truth} = c \land \text{pred} = c$
     - False Positives ($FP_c$): $\text{truth} \ne c \land \text{pred} = c$
     - False Negatives ($FN_c$): $\text{truth} = c \land \text{pred} \ne c$
   - $\text{IoU}(c) = \frac{TP_c}{TP_c + FP_c + FN_c}$.

### Equivalence Verification
Both approaches are mathematically equivalent, demonstrated by automated unit tests validating identical numerical outputs across all valid mask configurations.

---

## Stated Contract Fixtures & Expected Results

| Class | Ground Truth `[0, 1, 1, 0]` | Prediction `[0, 1, 0, 0]` | Intersection / $TP$ | Union / ($TP+FP+FN$) | Expected IoU |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **Class 0** | Indices $\{0, 3\}$ | Indices $\{0, 2, 3\}$ | $2$ (indices $\{0, 3\}$) | $3$ (indices $\{0, 2, 3\}$) | $\mathbf{2/3} \approx 0.6667$ |
| **Class 1** | Indices $\{1, 2\}$ | Indices $\{1\}$ | $1$ (index $\{1\}$) | $2$ (indices $\{1, 2\}$) | $\mathbf{1/2} = 0.5000$ |
| **Class 2** | Absent $\emptyset$ | Absent $\emptyset$ | $0$ | $0$ | $\mathbf{None}$ (`ABSENT_CLASS_SENTINEL`) |

---

## Scope & Limitations

- **Standard Library Only**: Uses pure Python standard library modules (`dataclasses`, `math`, `typing`); no external libraries (NumPy, PyTorch, Scikit-learn) required.
- **Flattened Fixtures**: Operates on 1D flattened representation of 2D/3D segmentation masks.
- **Hardware Agnostic**: Runs locally on CPU in any Python 3.8+ environment without requiring GPUs or cloud services.

---

## Execution Instructions

Run the concept example script:
```bash
python3 lessons/4-ComputerVision/12-Segmentation/applied-labs/r38-t033/example.py
```

Run the automated test suite:
```bash
python3 -m unittest discover -s lessons/4-ComputerVision/12-Segmentation/applied-labs/r38-t033 -p "test_*.py" -v
```
