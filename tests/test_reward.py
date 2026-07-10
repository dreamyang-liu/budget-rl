import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "reward/budget_probe_reward.py"
SPEC = importlib.util.spec_from_file_location("budget_probe_reward", MODULE_PATH)
reward = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reward)


class RewardTest(unittest.TestCase):
    def score(self, answer, ground_truth="[900, 1100]", remaining=1000):
        return reward.compute_score("test", answer, ground_truth, {"remaining_tokens": remaining})

    def test_tight_cover_gets_max_reward(self):
        self.assertAlmostEqual(self.score("<answer>[1000, 1000]</answer>"), 1.8)

    def test_wider_cover_is_penalized(self):
        self.assertAlmostEqual(self.score("<answer>[900, 1100]</answer>"), 1.44)

    def test_uncovered_interval_gets_zero(self):
        self.assertEqual(self.score("<answer>[1001, 1100]</answer>"), 0.0)

    def test_scalar_gets_zero(self):
        self.assertEqual(self.score("<answer>1000</answer>"), 0.0)

    def test_correct_impossible(self):
        self.assertEqual(self.score("<answer>impossible</answer>", "impossible"), 0.2)

    def test_invalid_format(self):
        self.assertEqual(self.score("[900, 1100]"), 0.0)

    def test_original_parser_accepts_embedded_interval(self):
        self.assertEqual(reward.parse_answer("<answer>junk [1, 2] junk</answer>"), (1, 2))


if __name__ == "__main__":
    unittest.main()
