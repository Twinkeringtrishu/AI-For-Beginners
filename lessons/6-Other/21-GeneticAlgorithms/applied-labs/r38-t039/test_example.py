"""
Unit tests for R38-T039: One genetic-selection round — scenario assessment.
Lesson: lessons/6-Other/21-GeneticAlgorithms
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from example import (
    TEACHING_DISCLAIMER,
    Candidate,
    SelectionResult,
    get_scenario_assessment,
    rank_by_fitness,
    select_elites,
)


class TestGeneticSelectionRound(unittest.TestCase):
    """Test suite for deterministic fitness ranking and elite selection."""

    def setUp(self):
        """Canonical fixture: A=3, B=8, C=5."""
        self.population = [
            Candidate("A", 3),
            Candidate("B", 8),
            Candidate("C", 5),
        ]

    def test_canonical_elite_selection_b_c(self):
        """Fitness A=3,B=8,C=5 with elite_count=2 selects B then C."""
        result = select_elites(self.population, 2)
        self.assertEqual(len(result.elites), 2)
        self.assertEqual(result.elites[0].candidate_id, "B")
        self.assertEqual(result.elites[1].candidate_id, "C")

    def test_canonical_non_elites(self):
        """With elite_count=2, non-elites contains only A."""
        result = select_elites(self.population, 2)
        self.assertEqual(len(result.non_elites), 1)
        self.assertEqual(result.non_elites[0].candidate_id, "A")

    def test_canonical_total_reward_one(self):
        """Elite B has fitness 8 (highest), C has fitness 5."""
        result = select_elites(self.population, 2)
        self.assertEqual(result.elites[0].fitness, 8)
        self.assertEqual(result.elites[1].fitness, 5)

    def test_rank_by_fitness_descending(self):
        """rank_by_fitness returns B, C, A in that order."""
        ranked = rank_by_fitness(self.population)
        ids = [c.candidate_id for c in ranked]
        self.assertEqual(ids, ["B", "C", "A"])

    def test_tie_broken_by_candidate_id(self):
        """Equal fitness: lower ID comes first (lexicographic)."""
        tied = [Candidate("Z", 5), Candidate("A", 5), Candidate("M", 5)]
        ranked = rank_by_fitness(tied)
        ids = [c.candidate_id for c in ranked]
        self.assertEqual(ids, ["A", "M", "Z"])

    def test_elite_count_zero(self):
        """elite_count=0 returns empty elites, all as non-elites."""
        result = select_elites(self.population, 0)
        self.assertEqual(result.elites, [])
        self.assertEqual(len(result.non_elites), 3)

    def test_elite_count_full_population(self):
        """elite_count == population size: all are elites, no non-elites."""
        result = select_elites(self.population, 3)
        self.assertEqual(len(result.elites), 3)
        self.assertEqual(result.non_elites, [])

    def test_elite_count_exceeds_population_raises(self):
        """elite_count > population size raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            select_elites(self.population, 4)
        self.assertIn("exceeds population size", str(cm.exception))

    def test_negative_elite_count_raises(self):
        """Negative elite_count raises ValueError."""
        with self.assertRaises(ValueError):
            select_elites(self.population, -1)

    def test_invalid_elite_count_type_raises(self):
        """Non-int elite_count (float, str, bool) raises TypeError."""
        for bad in (2.0, "2", True, None):
            with self.assertRaises(TypeError):
                select_elites(self.population, bad)

    def test_scenario_assessment_contains_key_phrases(self):
        """Scenario assessment report includes Q1, Q2, Q3 and disclaimer."""
        report = get_scenario_assessment(self.population, 2)
        self.assertIn("Q1", report)
        self.assertIn("Q2", report)
        self.assertIn("Q3", report)
        self.assertIn("Teaching approximation", report)
        self.assertIn("B", report)
        self.assertIn("C", report)

    def test_deliberate_wrong_result_fails(self):
        """A wrong expected elite list must not match the actual result."""
        result = select_elites(self.population, 2)
        actual_ids = [c.candidate_id for c in result.elites]
        wrong_ids = ["A", "C"]
        self.assertNotEqual(actual_ids, wrong_ids)
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_ids, wrong_ids)


if __name__ == "__main__":
    unittest.main()
