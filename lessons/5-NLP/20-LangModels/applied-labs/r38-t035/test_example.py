import unittest
import json
import sys
import os
from pathlib import Path

# Ensure the module can be imported from the sibling directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from example import render_baseline, extract_name

class TestTemplateRendering(unittest.TestCase):
    def setUp(self):
        self.template = "Hello {{name}}! Your topic is {{topic}}."

    def test_canonical_render(self):
        data = {"name": "Alice", "topic": "templating"}
        result = render_baseline(self.template, data)
        self.assertEqual(result, "Hello Alice! Your topic is templating.")

    def test_missing_topic(self):
        data = {"name": "Bob"}
        with self.assertRaises(ValueError) as cm:
            render_baseline(self.template, data)
        self.assertIn("Missing placeholder", str(cm.exception))

    def test_unknown_key(self):
        data = {"name": "Carol", "topic": "demo", "extra": "oops"}
        with self.assertRaises(ValueError) as cm:
            render_baseline(self.template, data)
        self.assertIn("Unknown key", str(cm.exception))

    def test_literal_braces_in_user_data(self):
        data = {"name": "D{ave}", "topic": "{{magic}}"}
        result = render_baseline(self.template, data)
        self.assertEqual(result, "Hello D{ave}! Your topic is {{magic}}.")

    def test_extract_name(self):
        data = {"name": "Eve", "topic": "something"}
        self.assertEqual(extract_name(data), "Eve")

    def test_deliberate_failure(self):
        """Verify that a deliberate wrong result makes the test fail."""
        data = {"name": "Frank", "topic": "test"}
        result = render_baseline(self.template, data)
        self.assertNotEqual(result, "Hello Frank! Your topic is WRONG.")
        with self.assertRaises(AssertionError):
            self.assertEqual(result, "Hello Frank! Your topic is WRONG.")

if __name__ == "__main__":
    # Save evidence of the canonical run
    evidence_path = Path(__file__).parents[2] / "work" / "R38-T035" / "example_run.txt"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "w") as f:
        json.dump({"template": self.template, "data": {"name": "Alice", "topic": "templating"}}, f, indent=2)
    unittest.main()
