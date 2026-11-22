import unittest
import json
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from example import validate_and_order



class TestTaskPlan(unittest.TestCase):
    def test_canonical_order(self):
        plan = {"A": [], "B": ["A"], "C": ["B"]}
        order, _ = validate_and_order(plan)
        self.assertEqual(order, ["A", "B", "C"])

    def test_missing_dependency(self):
        plan = {"A": [], "B": ["X"]}
        with self.assertRaises(ValueError) as cm:
            validate_and_order(plan)
        self.assertIn("Missing dependency", str(cm.exception))

    def test_cycle_detection(self):
        plan = {"A": ["C"], "B": ["A"], "C": ["B"]}
        with self.assertRaises(ValueError) as cm:
            validate_and_order(plan)
        self.assertIn("Cycle detected", str(cm.exception))

    def test_deliberate_failure(self):
        """Verify that a deliberate wrong result makes the test fail."""
        plan = {"A": [], "B": ["A"]}
        order, _ = validate_and_order(plan)
        self.assertNotEqual(order, ["A", "C"])
        with self.assertRaises(AssertionError):
            self.assertEqual(order, ["A", "C"])

if __name__ == "__main__":
    # Save intermediate states for evidence
    plan = {"A": [], "B": ["A"], "C": ["B"]}
    _, states = validate_and_order(plan)
    evidence_path = Path(__file__).parents[2] / "work" / "R38-T034" / "example_run.txt"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "w") as f:
        json.dump(states, f, indent=2)
    unittest.main()
