"""
Unit tests for R38-T038: Finite-state reward trace — evaluation experiment.
Lesson: lessons/6-Other/22-DeepRL
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from example import (
    DEFAULT_MAX_STEPS,
    TERMINAL_STATE,
    ThreeStateChainEnv,
    compare_three_configurations,
    get_evaluation_report,
    run_episode,
)


class TestFiniteStateRewardTrace(unittest.TestCase):
    """Test suite verifying 3-state chain MDP transitions, terminal rewards, step caps, and comparisons."""

    def test_canonical_right_right_ends_at_two_with_reward_one(self):
        """
        From state 0, [right, right] ends at 2 with total reward 1.
        """
        res = run_episode(["right", "right"], max_steps=5, start_state=0)
        self.assertEqual(res.final_state, 2)
        self.assertEqual(res.total_reward, 1)
        self.assertEqual(res.steps_taken, 2)
        self.assertTrue(res.reached_goal)
        self.assertFalse(res.truncated)

        # Inspect step-by-step history
        self.assertEqual(len(res.step_history), 2)
        self.assertEqual(res.step_history[0].state_before, 0)
        self.assertEqual(res.step_history[0].state_after, 1)
        self.assertEqual(res.step_history[0].reward, 0)

        self.assertEqual(res.step_history[1].state_before, 1)
        self.assertEqual(res.step_history[1].state_after, 2)
        self.assertEqual(res.step_history[1].reward, 1)
        self.assertTrue(res.step_history[1].terminated)

    def test_canonical_repeated_left_ends_at_step_cap_with_reward_zero(self):
        """
        Repeated left ends at the step cap with reward 0.
        """
        res = run_episode(["left"] * 5, max_steps=5, start_state=0)
        self.assertEqual(res.final_state, 0)
        self.assertEqual(res.total_reward, 0)
        self.assertEqual(res.steps_taken, 5)
        self.assertFalse(res.reached_goal)
        self.assertTrue(res.truncated)

    def test_single_step_right_moves_to_one_no_reward(self):
        """A single 'right' from 0 reaches 1 with 0 reward."""
        res = run_episode(["right"], max_steps=5, start_state=0)
        self.assertEqual(res.final_state, 1)
        self.assertEqual(res.total_reward, 0)
        self.assertEqual(res.steps_taken, 1)
        self.assertFalse(res.reached_goal)
        self.assertFalse(res.truncated)

    def test_left_boundary_clamped_at_zero(self):
        """Action 'left' at state 0 stays at state 0."""
        env = ThreeStateChainEnv(start_state=0)
        next_state, reward, terminated = env.step("left")
        self.assertEqual(next_state, 0)
        self.assertEqual(reward, 0)
        self.assertFalse(terminated)

    def test_backtracking_path(self):
        """Path ['right', 'left', 'right', 'right'] reaches goal at step 4."""
        res = run_episode(["right", "left", "right", "right"], max_steps=5, start_state=0)
        self.assertEqual(res.final_state, 2)
        self.assertEqual(res.total_reward, 1)
        self.assertEqual(res.steps_taken, 4)
        self.assertTrue(res.reached_goal)
        self.assertFalse(res.truncated)

    def test_episode_terminates_immediately_at_state_two(self):
        """Excess actions after reaching terminal state are ignored."""
        res = run_episode(["right", "right", "right", "right", "right"], max_steps=5, start_state=0)
        self.assertEqual(res.final_state, 2)
        self.assertEqual(res.steps_taken, 2)
        self.assertEqual(res.total_reward, 1)

    def test_custom_step_cap_truncation(self):
        """Capping episode at 3 steps cuts off a 4-step sequence."""
        res = run_episode(["right", "left", "right", "right"], max_steps=3, start_state=0)
        self.assertEqual(res.steps_taken, 3)
        self.assertEqual(res.final_state, 1)
        self.assertEqual(res.total_reward, 0)
        self.assertFalse(res.reached_goal)
        self.assertTrue(res.truncated)

    def test_invalid_action_raises_value_error(self):
        """Unrecognized actions raise ValueError."""
        with self.assertRaises(ValueError):
            run_episode(["jump"])
        with self.assertRaises(ValueError):
            run_episode(["up"])

    def test_invalid_action_type_raises_type_error(self):
        """Non-string actions raise TypeError."""
        with self.assertRaises(TypeError):
            run_episode([123])  # type: ignore
        with self.assertRaises(TypeError):
            run_episode([None])  # type: ignore

    def test_invalid_step_cap_raises_value_error(self):
        """Non-positive max_steps raise ValueError."""
        with self.assertRaises(ValueError):
            run_episode(["right"], max_steps=0)
        with self.assertRaises(ValueError):
            run_episode(["right"], max_steps=-2)

    def test_three_configurations_comparison_and_tradeoffs(self):
        """
        Focus: Compare three small controlled configurations using sample efficiency
        and goal completion metrics.
        """
        comparisons = compare_three_configurations()
        self.assertEqual(len(comparisons), 3)

        cfg_a, cfg_b, cfg_c = comparisons

        # Config A: Optimal greedy
        self.assertEqual(cfg_a.config_name, "Config-A (Optimal Greedy)")
        self.assertEqual(cfg_a.steps_taken, 2)
        self.assertEqual(cfg_a.total_reward, 1)
        self.assertTrue(cfg_a.reached_goal)
        self.assertAlmostEqual(cfg_a.sample_efficiency, 0.500, places=3)

        # Config B: Hesitant exploration
        self.assertEqual(cfg_b.config_name, "Config-B (Hesitant / Exploration)")
        self.assertEqual(cfg_b.steps_taken, 4)
        self.assertEqual(cfg_b.total_reward, 1)
        self.assertTrue(cfg_b.reached_goal)
        self.assertAlmostEqual(cfg_b.sample_efficiency, 0.250, places=3)

        # Config C: Left-biased adversarial
        self.assertEqual(cfg_c.config_name, "Config-C (Left-Biased / Adversarial)")
        self.assertEqual(cfg_c.steps_taken, 5)
        self.assertEqual(cfg_c.total_reward, 0)
        self.assertFalse(cfg_c.reached_goal)
        self.assertAlmostEqual(cfg_c.sample_efficiency, 0.000, places=3)

        # Verify structured report formatting
        report = get_evaluation_report(comparisons)
        self.assertIn("R38-T038: Three Controlled Configurations", report)
        self.assertIn("Config-A", report)
        self.assertIn("Config-B", report)
        self.assertIn("Config-C", report)

    def test_deliberate_wrong_result_fails(self):
        """
        Verify that a deliberate wrong result makes the test fail.
        Done When criterion: 'a deliberate wrong result makes the test fail'.
        """
        res = run_episode(["right", "right"])
        actual_reward = res.total_reward  # 1
        wrong_reward = 0

        self.assertNotEqual(actual_reward, wrong_reward)
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_reward, wrong_reward)


if __name__ == "__main__":
    unittest.main()
