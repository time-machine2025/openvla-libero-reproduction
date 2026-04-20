from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openvla_libero_reproduction.dataset import generate_eval_episodes, load_episodes
from openvla_libero_reproduction.evaluate import evaluate_policy
from openvla_libero_reproduction.policy import BehaviorCloningPolicy


class PipelineTest(unittest.TestCase):
    def test_behavior_cloning_beats_random_free_fall(self) -> None:
        train_episodes = load_episodes(ROOT / "data" / "train_episodes.jsonl")
        eval_episodes = list(generate_eval_episodes())
        policy = BehaviorCloningPolicy().fit(train_episodes)
        metrics = evaluate_policy(policy, eval_episodes)
        self.assertGreaterEqual(metrics["success_rate"], 0.8)


if __name__ == "__main__":
    unittest.main()
