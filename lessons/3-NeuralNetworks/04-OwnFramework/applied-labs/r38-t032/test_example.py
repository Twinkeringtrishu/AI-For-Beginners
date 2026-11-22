"""
Unit tests for R38-T032: Single perceptron update — boundary-case lesson extension.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from example import (
    BOUNDARY_CASE_CATALOG,
    PerceptronState,
    PerceptronStepResult,
    compute_score,
    perceptron_update_step,
    predict,
    predict_score,
)


class TestSinglePerceptronUpdate(unittest.TestCase):
    """Verify single perceptron prediction and update contracts including all boundary cases."""

    def test_canonical_stated_scenario(self):
        """
        Stated scenario:
        Weights [0,0], bias 0, input [1,2], target 1, initial prediction 0:
        learning-rate 1 update gives weights [1,2], bias 1.
        """
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0, learning_rate=1.0)
        res = perceptron_update_step(state, [1, 2], target=1)

        # Initial forward pass verification
        self.assertEqual(res.initial_score, 0.0)
        self.assertEqual(res.initial_prediction, 0)
        self.assertEqual(res.target, 1)
        self.assertEqual(res.error, 1)
        self.assertEqual(res.delta, 1.0)

        # Updated state verification
        self.assertEqual(res.new_weights, (1.0, 2.0))
        self.assertEqual(res.new_bias, 1.0)
        self.assertTrue(res.weights_updated)

    def test_zero_score_tie_rule_predicts_zero(self):
        """Documented zero-score tie rule: exact zero score breaks ties toward 0."""
        self.assertEqual(predict(0.0), 0)
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0)
        score, pred = predict_score(state, [10.0, 20.0])
        self.assertEqual(score, 0.0)
        self.assertEqual(pred, 0)

    def test_near_zero_boundary_decisions(self):
        """Test boundary threshold decisions for positive and negative epsilons."""
        self.assertEqual(predict(1e-12), 1)
        self.assertEqual(predict(-1e-12), 0)

    def test_correct_prediction_no_update(self):
        """When prediction matches target (error=0), weights and bias remain unchanged."""
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0, learning_rate=1.0)
        # score is 0.0 -> prediction is 0. Target is 0 -> error is 0.
        res = perceptron_update_step(state, [1, 2], target=0)

        self.assertEqual(res.initial_prediction, 0)
        self.assertEqual(res.error, 0)
        self.assertEqual(res.delta, 0.0)
        self.assertEqual(res.new_weights, (0.0, 0.0))
        self.assertEqual(res.new_bias, 0.0)
        self.assertFalse(res.weights_updated)

    def test_negative_error_update(self):
        """When target is 0 and prediction is 1, error is -1, reducing weights and bias."""
        state = PerceptronState(weights=(1.0, 1.0), bias=1.0, learning_rate=0.5)
        # score = 1*1 + 1*1 + 1 = 3.0 > 0 -> prediction = 1
        # error = 0 - 1 = -1 -> delta = 0.5 * -1 = -0.5
        res = perceptron_update_step(state, [1.0, 1.0], target=0)

        self.assertEqual(res.initial_prediction, 1)
        self.assertEqual(res.error, -1)
        self.assertEqual(res.delta, -0.5)
        self.assertEqual(res.new_weights, (0.5, 0.5))
        self.assertEqual(res.new_bias, 0.5)
        self.assertTrue(res.weights_updated)

    def test_catalog_boundary_cases_pass(self):
        """Verify that every catalogued boundary case passes its declared expectations."""
        for case in BOUNDARY_CASE_CATALOG:
            state = PerceptronState(weights=case["weights"], bias=case["bias"], learning_rate=case["lr"])
            res = perceptron_update_step(state, case["input"], case["target"])
            self.assertAlmostEqual(res.initial_score, case["expected_score"], places=7)
            self.assertEqual(res.initial_prediction, case["expected_pred"])
            self.assertEqual(res.new_weights, case["expected_weights"])
            self.assertEqual(res.new_bias, case["expected_bias"])

    def test_dimension_mismatch_raises_value_error(self):
        """Input vectors with length != 2 must raise ValueError."""
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0)
        with self.assertRaises(ValueError):
            perceptron_update_step(state, [1], target=1)
        with self.assertRaises(ValueError):
            perceptron_update_step(state, [1, 2, 3], target=1)

    def test_empty_input_raises_value_error(self):
        """Empty input vector raises ValueError."""
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0)
        with self.assertRaises(ValueError):
            perceptron_update_step(state, [], target=1)

    def test_non_binary_target_raises_value_error(self):
        """Non-binary integer targets (e.g. 2, -1) must raise ValueError."""
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0)
        with self.assertRaises(ValueError):
            perceptron_update_step(state, [1, 2], target=2)
        with self.assertRaises(ValueError):
            perceptron_update_step(state, [1, 2], target=-1)

    def test_invalid_target_type_raises_type_error(self):
        """Non-integer targets (floats, booleans, strings) must raise TypeError."""
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0)
        with self.assertRaises(TypeError):
            perceptron_update_step(state, [1, 2], target=0.5)  # type: ignore
        with self.assertRaises(TypeError):
            perceptron_update_step(state, [1, 2], target=True)  # type: ignore
        with self.assertRaises(TypeError):
            perceptron_update_step(state, [1, 2], target="1")  # type: ignore

    def test_non_numeric_input_raises_type_error(self):
        """Inputs containing non-numeric values must raise TypeError."""
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0)
        with self.assertRaises(TypeError):
            perceptron_update_step(state, [1, "two"], target=1)  # type: ignore
        with self.assertRaises(TypeError):
            perceptron_update_step(state, [1, None], target=1)  # type: ignore
        with self.assertRaises(TypeError):
            perceptron_update_step(state, [True, False], target=1)  # type: ignore

    def test_non_finite_input_raises_value_error(self):
        """Inputs containing NaN or Infinity must raise ValueError."""
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0)
        with self.assertRaises(ValueError):
            perceptron_update_step(state, [float("nan"), 2.0], target=1)
        with self.assertRaises(ValueError):
            perceptron_update_step(state, [1.0, float("inf")], target=1)

    def test_invalid_learning_rate_raises_error(self):
        """Learning rates <= 0 or non-finite must raise ValueError."""
        with self.assertRaises(ValueError):
            perceptron_update_step(PerceptronState(weights=(0.0, 0.0), bias=0.0, learning_rate=0.0), [1, 2], 1)
        with self.assertRaises(ValueError):
            perceptron_update_step(PerceptronState(weights=(0.0, 0.0), bias=0.0, learning_rate=-1.0), [1, 2], 1)
        with self.assertRaises(ValueError):
            perceptron_update_step(PerceptronState(weights=(0.0, 0.0), bias=0.0, learning_rate=float("inf")), [1, 2], 1)

    def test_deliberate_wrong_result_fails(self):
        """
        Verify that a deliberate wrong result makes the test fail.
        Done When criterion: 'a deliberate wrong result makes the test fail'.
        """
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0, learning_rate=1.0)
        res = perceptron_update_step(state, [1, 2], target=1)

        # Actual updated weights are (1.0, 2.0); expecting unchanged weights fails
        self.assertNotEqual(res.new_weights, (0.0, 0.0))
        with self.assertRaises(AssertionError):
            self.assertEqual(res.new_weights, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
