from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

Position = Tuple[int, int]


@dataclass
class Observation:
    instruction: str
    agent: Position
    object_pos: Position
    goal: Position
    carrying: bool
    steps_taken: int
    max_steps: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "instruction": self.instruction,
            "agent": list(self.agent),
            "object": list(self.object_pos),
            "goal": list(self.goal),
            "carrying": self.carrying,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
        }


class GridManipEnv:
    ACTIONS = ("up", "down", "left", "right", "pick", "place")

    def __init__(
        self,
        instruction: str,
        agent: Position,
        object_pos: Position,
        goal: Position,
        width: int = 4,
        height: int = 4,
        max_steps: int = 20,
    ) -> None:
        self.instruction = instruction
        self.agent = agent
        self.object_pos = object_pos
        self.goal = goal
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.carrying = False
        self.steps_taken = 0

    def clone(self) -> "GridManipEnv":
        return GridManipEnv(
            instruction=self.instruction,
            agent=self.agent,
            object_pos=self.object_pos,
            goal=self.goal,
            width=self.width,
            height=self.height,
            max_steps=self.max_steps,
        )

    def observation(self) -> Observation:
        return Observation(
            instruction=self.instruction,
            agent=self.agent,
            object_pos=self.object_pos,
            goal=self.goal,
            carrying=self.carrying,
            steps_taken=self.steps_taken,
            max_steps=self.max_steps,
        )

    def done(self) -> bool:
        return self.success() or self.steps_taken >= self.max_steps

    def success(self) -> bool:
        return self.object_pos == self.goal and not self.carrying

    def step(self, action: str) -> Observation:
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        if self.done():
            return self.observation()

        self.steps_taken += 1
        x, y = self.agent
        if action == "up":
            self.agent = (x, max(0, y - 1))
        elif action == "down":
            self.agent = (x, min(self.height - 1, y + 1))
        elif action == "left":
            self.agent = (max(0, x - 1), y)
        elif action == "right":
            self.agent = (min(self.width - 1, x + 1), y)
        elif action == "pick" and self.agent == self.object_pos and not self.carrying:
            self.carrying = True
        elif action == "place" and self.carrying:
            self.carrying = False
            self.object_pos = self.agent

        if self.carrying:
            self.object_pos = self.agent
        return self.observation()
