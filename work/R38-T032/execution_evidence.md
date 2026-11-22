# Execution Evidence for Issue #32 (R38-T032)

- **Student**: Tanishka .
- **Task**: R38-T032: single perceptron update — boundary-case lesson extension [lessons/3-NeuralNetworks/04-OwnFramework]
- **Repository**: https://github.com/S-AI-Projects/AI-For-Beginners
- **Branch**: `R38-T032`
- **Lab Directory**: `lessons/3-NeuralNetworks/04-OwnFramework/applied-labs/r38-t032`
- **Environment**: Python 3.15.2, git version 2.55.0

---

## 1. Automated Test Suite Execution

```bash
python3 -m unittest discover -s lessons/3-NeuralNetworks/04-OwnFramework/applied-labs/r38-t032 -p "test_*.py" -v
```

### Output:
```
test_canonical_stated_scenario (test_example.TestSinglePerceptronUpdate.test_canonical_stated_scenario)
Stated scenario: ... ok
test_catalog_boundary_cases_pass (test_example.TestSinglePerceptronUpdate.test_catalog_boundary_cases_pass)
Verify that every catalogued boundary case passes its declared expectations. ... ok
test_correct_prediction_no_update (test_example.TestSinglePerceptronUpdate.test_correct_prediction_no_update)
When prediction matches target (error=0), weights and bias remain unchanged. ... ok
test_deliberate_wrong_result_fails (test_example.TestSinglePerceptronUpdate.test_deliberate_wrong_result_fails)
Verify that a deliberate wrong result makes the test fail. ... ok
test_dimension_mismatch_raises_value_error (test_example.TestSinglePerceptronUpdate.test_dimension_mismatch_raises_value_error)
Input vectors with length != 2 must raise ValueError. ... ok
test_empty_input_raises_value_error (test_example.TestSinglePerceptronUpdate.test_empty_input_raises_value_error)
Empty input vector raises ValueError. ... ok
test_invalid_learning_rate_raises_error (test_example.TestSinglePerceptronUpdate.test_invalid_learning_rate_raises_error)
Learning rates <= 0 or non-finite must raise ValueError. ... ok
test_invalid_target_type_raises_type_error (test_example.TestSinglePerceptronUpdate.test_invalid_target_type_raises_type_error)
Non-integer targets (floats, booleans, strings) must raise TypeError. ... ok
test_near_zero_boundary_decisions (test_example.TestSinglePerceptronUpdate.test_near_zero_boundary_decisions)
Test boundary threshold decisions for positive and negative epsilons. ... ok
test_negative_error_update (test_example.TestSinglePerceptronUpdate.test_negative_error_update)
When target is 0 and prediction is 1, error is -1, reducing weights and bias. ... ok
test_non_binary_target_raises_value_error (test_example.TestSinglePerceptronUpdate.test_non_binary_target_raises_value_error)
Non-binary integer targets (e.g. 2, -1) must raise ValueError. ... ok
test_non_finite_input_raises_value_error (test_example.TestSinglePerceptronUpdate.test_non_finite_input_raises_value_error)
Inputs containing NaN or Infinity must raise ValueError. ... ok
test_non_numeric_input_raises_type_error (test_example.TestSinglePerceptronUpdate.test_non_numeric_input_raises_type_error)
Inputs containing non-numeric values must raise TypeError. ... ok
test_zero_score_tie_rule_predicts_zero (test_example.TestSinglePerceptronUpdate.test_zero_score_tie_rule_predicts_zero)
Documented zero-score tie rule: exact zero score breaks ties toward 0. ... ok

----------------------------------------------------------------------
Ran 14 tests in 0.001s

OK
```

---

## 2. Example Execution Output

```bash
python3 lessons/3-NeuralNetworks/04-OwnFramework/applied-labs/r38-t032/example.py
```

### Output:
```
======================================================================
R38-T032: Single Perceptron Update Boundary-Case Lesson Extension
======================================================================

--- 1. Canonical Stated Contract Execution ---
Initial State : weights=(0.0, 0.0), bias=0.0, learning_rate=1.0
Input Sample  : x=[1, 2], target=1
Forward Pass  : score = 0.0000
Tie Rule Dec. : score <= 0 -> prediction = 0
Error & Delta : error = target - pred = 1, delta = 1.0
Updated State : weights = (1.0, 2.0), bias = 1.0

--- 2. Boundary Case Scenarios ---
[Smallest Valid (Canonical)] Initial zero weights with target=1: score=0 -> pred=0 -> update to weights [1,2], bias 1
  Score: 0.0 -> Pred: 0 (Target: 1)
  New Weights: (1.0, 2.0), New Bias: 1.0
[Boundary (Exact Zero Tie)] Zero score with target=0: score=0 -> pred=0 -> target=0, zero error, NO update
  Score: 0.0 -> Pred: 0 (Target: 0)
  New Weights: (0.0, 0.0), New Bias: 0.0
[Boundary (Near Zero Positive)] Score slightly positive (+1e-9): score > 0 -> pred=1
  Score: 1e-09 -> Pred: 1 (Target: 1)
  New Weights: (1e-09, 0.0), New Bias: 0.0
[Boundary (Near Zero Negative)] Score slightly negative (-1e-9): score <= 0 -> pred=0
  Score: -1e-09 -> Pred: 0 (Target: 0)
  New Weights: (-1e-09, 0.0), New Bias: 0.0
[Boundary (Negative Update Direction)] Target=0, prediction=1: error=-1, weights decrease by lr*x
  Score: 3.0 -> Pred: 1 (Target: 0)
  New Weights: (0.5, 0.0), New Bias: -0.5
```

---

## 3. Deliberate Failure Verification

Running a test case with an incorrect expectation exits with code 1:
```
AssertionError: (1.0, 2.0) != (0.0, 0.0)
FAILED (failures=1)
Exit code: 1
```
