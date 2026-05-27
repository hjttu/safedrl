from dataclasses import dataclass
from typing import Dict, Hashable, Tuple

import numpy as np


@dataclass
class _MemoryEntry:
    value: float
    invisible_steps: int = 0


class TemporalResponsibilityMemory:
    """Risk-adaptive exponential memory for pairwise responsibility."""

    def __init__(
        self,
        beta_min: float = 0.2,
        beta_max: float = 0.95,
        reset_steps: int = 20,
        reset_h: float = 1.0,
        decay: float = 0.9,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.reset_steps = reset_steps
        self.reset_h = reset_h
        self.decay = decay
        self.memory: Dict[Hashable, _MemoryEntry] = {}

    @staticmethod
    def agent_pair_key(env_id: int, i: int, j: int) -> Tuple[int, int, int]:
        a, b = sorted((int(i), int(j)))
        return int(env_id), a, b

    def update(self, key: Hashable, gamma_hat: float, risk_score: float, h_ij: float | None = None) -> float:
        gamma_hat = float(np.clip(gamma_hat, 0.0, 1.0))
        risk_score = float(np.clip(risk_score, 0.0, 1.0))
        if h_ij is not None and h_ij > self.reset_h:
            self.memory.pop(key, None)
            return gamma_hat

        prev = self.memory.get(key, _MemoryEntry(gamma_hat)).value
        beta = self.beta_min + (self.beta_max - self.beta_min) * risk_score
        gamma = float(np.clip(beta * prev + (1.0 - beta) * gamma_hat, 0.0, 1.0))
        self.memory[key] = _MemoryEntry(gamma, 0)
        return gamma

    def update_agent_pair(
        self,
        env_id: int,
        i: int,
        j: int,
        gamma_hat_ij: float,
        risk_score: float,
        h_ij: float | None = None,
    ) -> tuple[float, float]:
        key = self.agent_pair_key(env_id, i, j)
        low_i, _ = sorted((int(i), int(j)))
        gamma_for_low = self.update(key, gamma_hat_ij if int(i) == low_i else 1.0 - gamma_hat_ij, risk_score, h_ij)
        gamma_ij = gamma_for_low if int(i) == low_i else 1.0 - gamma_for_low
        return float(np.clip(gamma_ij, 0.0, 1.0)), float(np.clip(1.0 - gamma_ij, 0.0, 1.0))

    def decay_unseen(self, visible_keys) -> None:
        visible = set(visible_keys)
        for key in list(self.memory.keys()):
            if key in visible:
                continue
            entry = self.memory[key]
            entry.invisible_steps += 1
            entry.value = 0.5 + self.decay * (entry.value - 0.5)
            if entry.invisible_steps >= self.reset_steps:
                self.memory.pop(key, None)
