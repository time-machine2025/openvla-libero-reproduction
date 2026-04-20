from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Protocol

from .dataset import Episode
from .env import GridManipEnv, Observation


def _sign(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def make_feature(observation: Observation) -> tuple[object, ...]:
    ax, ay = observation.agent
    ox, oy = observation.object_pos
    gx, gy = observation.goal
    return (
        observation.instruction,
        _sign(ox - ax),
        _sign(oy - ay),
        _sign(gx - ax),
        _sign(gy - ay),
        observation.carrying,
    )


class Policy(Protocol):
    name: str

    def act(self, observation: Observation) -> str:
        ...


@dataclass
class HeuristicPolicy:
    name: str = "heuristic"

    def act(self, observation: Observation) -> str:
        ax, ay = observation.agent
        ox, oy = observation.object_pos
        gx, gy = observation.goal
        target_x, target_y = (gx, gy) if observation.carrying else (ox, oy)
        if (ax, ay) == (target_x, target_y):
            return "place" if observation.carrying else "pick"
        if ax < target_x:
            return "right"
        if ax > target_x:
            return "left"
        if ay < target_y:
            return "down"
        return "up"


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, observation: Observation) -> str:
        del observation
        return self.rng.choice(GridManipEnv.ACTIONS)


class BehaviorCloningPolicy:
    name = "behavior_cloning"

    def __init__(self) -> None:
        self.lookup: Dict[tuple[object, ...], str] = {}
        self.fallback = HeuristicPolicy()

    def fit(self, episodes: Iterable[Episode], max_steps: int = 20) -> "BehaviorCloningPolicy":
        action_counts: Dict[tuple[object, ...], Counter[str]] = defaultdict(Counter)
        for episode in episodes:
            env = episode.build_env(max_steps=max_steps)
            for action in episode.actions:
                feature = make_feature(env.observation())
                action_counts[feature][action] += 1
                env.step(action)

        self.lookup = {
            feature: counts.most_common(1)[0][0]
            for feature, counts in action_counts.items()
        }
        return self

    def act(self, observation: Observation) -> str:
        feature = make_feature(observation)
        if feature in self.lookup:
            return self.lookup[feature]
        return self.fallback.act(observation)
