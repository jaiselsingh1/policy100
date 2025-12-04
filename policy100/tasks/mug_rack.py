"""Task definition for hanging a mug on a rack.

This module defines a simple reward function and reset logic for the mug‑on‑rack task.  The
`MugRackTask` class computes a reward based on the distance between the mug’s handle and
the target hook and whether the mug has been lifted off the table.  It is deliberately
lightweight and does not implement domain randomisation or complex shaping.
"""

from __future__ import annotations

import mujoco
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MugRackConfig:
    """Configuration for the mug‑on‑rack task."""
    name: str = "mug rack"
    tol_xy: float = 0.05  # success tolerance in meters
    reach_alpha: float = 4.0  # exponential scaling for distance penalty
    lift_height: float = 0.02  # minimum height above table to count as lifted
    lift_bonus: float = 0.25  # bonus added when mug is lifted


class MugRackTask:
    def __init__(self, env, config: Optional[MugRackConfig] = None):
        self.env = env
        self.cfg = config or MugRackConfig()
        # cache site ids
        self._sid_handle = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "handle_hole")
        self._sid_hook = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "hook_2_target")
        # free joint addresses for the mug
        self._qadr_mug, self._dadr_mug = env._qadr_mug, env._dadr_mug

    def reset(self, seed: Optional[int] = None, randomize: bool = False):
        """Spawn the mug at its initial location.

        If `randomize` is True, the mug’s x/y position will be jittered by up to ±10 cm.
        """
        rng = np.random.default_rng(seed)
        # base relative position: slightly in front of the robot mount on the table
        rel = np.array([-0.10, 0.0, 0.44 - self.env.table_z], dtype=np.float64)
        if randomize:
            rel[:2] += rng.uniform(-0.10, 0.10, size=2)
        # convert to world coordinates
        world = self.env._table_to_world(rel)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._spawn(self._qadr_mug, self._dadr_mug, xyz=world, quat=quat)

    def _spawn(self, qadr: int, dadr: int, *, xyz: np.ndarray, quat: np.ndarray) -> None:
        q = np.asarray(quat, dtype=np.float64)
        q /= max(np.linalg.norm(q), 1e-9)
        p = np.asarray(xyz, dtype=np.float64)
        self.env.data.qpos[qadr:qadr+4] = q
        self.env.data.qpos[qadr+4:qadr+7] = p
        self.env.data.qvel[dadr:dadr+6] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

    def _metrics(self) -> dict[str, float | np.ndarray | bool]:
        d = self.env.data
        handle_pos = d.site_xpos[self._sid_handle].copy()
        hook_pos = d.site_xpos[self._sid_hook].copy()
        # z coordinate of the mug (xyz is after quaternion in free joint state)
        mug_z = d.qpos[self._qadr_mug + 6]
        d_hang = float(np.linalg.norm(handle_pos - hook_pos))
        lifted = bool(mug_z > (self.env.table_z + self.cfg.lift_height))
        return {"handle_pos": handle_pos, "hook_pos": hook_pos, "d_hang": d_hang, "lifted": lifted}

    def reward(self) -> tuple[float, dict]:
        """Compute the reward and an info dict.

        Reward decays exponentially with the distance between the mug handle and target hook.
        A lift bonus is added when the mug is raised above the table.  Success is flagged
        when the handle is within `tol_xy` metres of the hook and the mug is lifted.
        """
        m = self._metrics()
        r = np.exp(-self.cfg.reach_alpha * m["d_hang"])
        if m["lifted"]:
            r += self.cfg.lift_bonus
        success = (m["d_hang"] < self.cfg.tol_xy) and m["lifted"]
        info = {"success": bool(success), **m}
        return float(r), info

    def cost(self) -> float:
        r, _ = self.reward()
        return -float(r)