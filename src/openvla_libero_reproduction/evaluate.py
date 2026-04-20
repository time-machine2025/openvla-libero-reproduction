from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from .dataset import Episode
from .policy import Policy


@dataclass
class RolloutResult:
    success: bool
    steps: int


def run_episode(policy: Policy, episode: Episode, max_steps: int = 20) -> RolloutResult:
    env = episode.build_env(max_steps=max_steps)
    while not env.done():
        action = policy.act(env.observation())
        env.step(action)
    return RolloutResult(success=env.success(), steps=env.steps_taken)


def evaluate_policy(policy: Policy, episodes: Iterable[Episode], max_steps: int = 20) -> Dict[str, object]:
    rollouts: List[RolloutResult] = [run_episode(policy, episode, max_steps=max_steps) for episode in episodes]
    return {
        "policy": policy.name,
        "episodes": len(rollouts),
        "success_rate": sum(int(item.success) for item in rollouts) / max(1, len(rollouts)),
        "avg_steps": mean(item.steps for item in rollouts) if rollouts else 0.0,
    }


def save_metrics(path: str | Path, payload: Dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
