# tb_logger.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


@dataclass
class TBLogger:
    """
    A lightweight TensorBoard logger for RL training.

    - Uses `global_step` as x-axis (recommended: env timesteps).
    - Writes to: {root}/{exp_name}/{run_name}
    """
    root: str = "runs"
    exp_name: str = "experiment"
    run_name: Optional[str] = None
    flush_secs: int = 10

    def __post_init__(self) -> None:
        if self.run_name is None:
            self.run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logdir = os.path.join(self.root, self.exp_name, self.run_name)
        os.makedirs(self.logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.logdir, flush_secs=self.flush_secs)

    def add_hparams(self, hparams: Dict[str, Any], metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Optional: record hyperparameters in TensorBoard (HParams plugin).
        Note: TB hparams needs scalar metrics; if you don't have them yet, pass {}.
        """
        metrics = metrics or {}
        # TensorBoard expects hparams values to be int/float/str/bool
        safe_hparams: Dict[str, Any] = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                safe_hparams[k] = v
            else:
                safe_hparams[k] = str(v)
        self.writer.add_hparams(safe_hparams, metrics)

    def log_episode(self, episode_reward: float, episode_len: int, episode_idx: int, global_step: int) -> None:
        # x-axis: env timesteps
        self.writer.add_scalar("charts/episode_reward", episode_reward, global_step)
        self.writer.add_scalar("charts/episode_length", episode_len, global_step)
        # also log episode index (useful for debugging)
        self.writer.add_scalar("charts/episode_idx", episode_idx, global_step)

    def log_avg_reward(self, avg_reward: float, global_step: int) -> None:
        self.writer.add_scalar("charts/avg_reward", avg_reward, global_step)

    def log_loss_dict(self, losses: Dict[str, float], global_step: int) -> None:
        """
        Call this if/when your PPO.update() returns loss metrics.
        Example dict: {"total": 1.2, "actor": 0.3, "critic": 0.9}
        """
        for k, v in losses.items():
            self.writer.add_scalar(f"loss/{k}", float(v), global_step)

    def close(self) -> None:
        self.writer.close()
