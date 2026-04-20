from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .env import GridManipEnv


@dataclass(frozen=True)
class Episode:
    instruction: str
    agent: tuple[int, int]
    object_pos: tuple[int, int]
    goal: tuple[int, int]
    actions: tuple[str, ...]

    def build_env(self, max_steps: int = 20) -> GridManipEnv:
        return GridManipEnv(
            instruction=self.instruction,
            agent=self.agent,
            object_pos=self.object_pos,
            goal=self.goal,
            max_steps=max_steps,
        )


def load_episodes(path: str | Path) -> List[Episode]:
    episodes: List[Episode] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        episodes.append(
            Episode(
                instruction=str(payload["instruction"]),
                agent=tuple(payload["agent"]),
                object_pos=tuple(payload["object"]),
                goal=tuple(payload["goal"]),
                actions=tuple(payload["actions"]),
            )
        )
    return episodes


def generate_eval_episodes() -> Iterable[Episode]:
    seeds = [
        ("pick the red block and place it on the left goal", (1, 0), (3, 1), (0, 3)),
        ("pick the red block and place it on the left goal", (2, 3), (2, 0), (0, 3)),
        ("pick the blue cup and place it on the right tray", (0, 0), (0, 2), (3, 0)),
        ("pick the blue cup and place it on the right tray", (2, 2), (1, 0), (3, 0)),
        ("pick the green bottle and place it on the top shelf", (3, 3), (1, 2), (2, 0)),
        ("pick the green bottle and place it on the top shelf", (0, 1), (2, 1), (2, 0)),
    ]
    for instruction, agent, object_pos, goal in seeds:
        yield Episode(
            instruction=instruction,
            agent=agent,
            object_pos=object_pos,
            goal=goal,
            actions=(),
        )
