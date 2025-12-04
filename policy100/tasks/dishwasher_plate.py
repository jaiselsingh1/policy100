"""Task definition for inserting a plate into a dishwasher rack.

This module defines a simple reward function for the plate insertion task.  The reward is
based on the distance between the plate centre and the target slot, with a bonus for
lifting the plate off the table.  The implementation is intended to be clear and
lightweight, serving as a starting point for further experimentation.
"""

from __future__ import annotations

import mujoco
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DishwasherPlateConfig:
    """Configuration for the dishwasher plate task."""
    name: str = "dishwasher plate"
    tol_xy: float = 0.05  # success tolerance
    reach_alpha: float = 4.0  # exponential distance scaling
    lift_height: float = 0.02  # minimum height above table to count as lifted
    lift_bonus: float = 0.25  # bonus for lifting


class DishwasherPlateTask:
    def __init__(self, env, config: Optional[DishwasherPlateConfig] = None):
        self.env = env
        self.cfg = config or DishwasherPlateConfig()
        self._sid_plate_center = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        self._sid_target = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "target_slot")
        self._qadr_plate, self._dadr_plate = env._qadr_plate, env._dadr_plate

    def reset(self, seed: Optional[int] = None, randomize: bool = False) -> None:
        rng = np.random.default_rng(seed)
        # Spawn plate on left side of the table
        rel = np.array([-0.15, 0.0, 0.43 - self.env.table_z], dtype=np.float64)
        if randomize:
            rel[:2] += rng.uniform(-0.10, 0.10, size=2)
        world = self.env._table_to_world(rel)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._spawn(self._qadr_plate, self._dadr_plate, xyz=world, quat=quat)

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
        plate_pos = d.site_xpos[self._sid_plate_center].copy()
        target_pos = d.site_xpos[self._sid_target].copy()
        plate_z = d.qpos[self._qadr_plate + 6]
        d_insert = float(np.linalg.norm(plate_pos - target_pos))
        lifted = bool(plate_z > (self.env.table_z + self.cfg.lift_height))
        return {"plate_pos": plate_pos, "target_pos": target_pos, "d_insert": d_insert, "lifted": lifted}

    def reward(self) -> tuple[float, dict]:
        m = self._metrics()
        r = np.exp(-self.cfg.reach_alpha * m["d_insert"])
        if m["lifted"]:
            r += self.cfg.lift_bonus
        success = (m["d_insert"] < self.cfg.tol_xy) and m["lifted"]
        info = {"success": bool(success), **m}
        return float(r), info

    def cost(self) -> float:
        r, _ = self.reward()
        return -float(r)