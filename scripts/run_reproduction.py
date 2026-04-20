from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openvla_libero_reproduction.dataset import generate_eval_episodes, load_episodes
from openvla_libero_reproduction.evaluate import evaluate_policy, save_metrics
from openvla_libero_reproduction.policy import BehaviorCloningPolicy, HeuristicPolicy, RandomPolicy


def main() -> None:
    root = ROOT
    config = json.loads((root / "configs" / "baseline.json").read_text(encoding="utf-8"))
    train_episodes = load_episodes(root / config["train_dataset"])
    eval_episodes = list(generate_eval_episodes())

    bc_policy = BehaviorCloningPolicy().fit(train_episodes, max_steps=config["max_steps"])
    results = {
        "behavior_cloning": evaluate_policy(bc_policy, eval_episodes, max_steps=config["max_steps"]),
        "heuristic": evaluate_policy(HeuristicPolicy(), eval_episodes, max_steps=config["max_steps"]),
        "random": evaluate_policy(RandomPolicy(seed=config["seed"]), eval_episodes, max_steps=config["max_steps"]),
    }
    output_path = root / "results" / "latest_metrics.json"
    save_metrics(output_path, results)

    print("Saved metrics to", output_path)
    for name, metrics in results.items():
        print(
            f"{name:>18} | success_rate={metrics['success_rate']:.2f} | "
            f"avg_steps={metrics['avg_steps']:.2f}"
        )


if __name__ == "__main__":
    main()
