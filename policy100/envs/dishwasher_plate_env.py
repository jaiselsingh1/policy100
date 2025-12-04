import mujoco
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces
from pathlib import Path

from ..tasks.dishwasher_plate import DishwasherPlateTask, DishwasherPlateConfig

class DishwasherPlateEnv(MujocoEnv):
    """
    MuJoCo environment for inserting a plate into a dishwasher rack.

    This environment loads the `plate_dishwasher_task.xml` asset, spawns a plate with a free joint
    and provides a continuous control interface.  Observations include joint positions/velocities
    and the plate pose.  Rewards are computed by the associated `DishwasherPlateTask`.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self,
                 model_path: str | None = None,
                 frame_skip: int = 5,
                 ctrl_scale: float = 0.01,
                 gr_ctrl_scale: float = 255.0,
                 include_tcp: bool = True,
                 include_rel: bool = True,
                 table_z: float = 0.38,
                 **kwargs):
        self.ctrl_scale = float(ctrl_scale)
        self.gr_ctrl_scale = float(gr_ctrl_scale)
        self.include_tcp = bool(include_tcp)
        self.include_rel = bool(include_rel)
        self.table_z = float(table_z)
        self.observation_space = None

        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent / "assets" / "plate_dishwasher_task.xml"
        else:
            model_path = Path(model_path)
        assert model_path.exists(), f"model XML not found: {model_path}"

        super().__init__(
            model_path=str(model_path),
            frame_skip=frame_skip,
            default_camera_config=None,
            observation_space=None,
            **kwargs
        )

        # Cache identifiers
        self._bid_plate = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        self._sid_plate_center = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        self._sid_target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_slot")

        self._qadr_plate, self._dadr_plate = self._freejoint_addr(self._bid_plate)

        if self.model.njnt > 0:
            try:
                self._arm_qpos_idx = self._find_arm_hinges(prefix="joint", count=7)
            except Exception:
                self._arm_qpos_idx = np.array([], dtype=np.int64)
        else:
            self._arm_qpos_idx = np.array([], dtype=np.int64)

        self.act_dim = len(self._arm_qpos_idx) + 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

        self.task = DishwasherPlateTask(self, DishwasherPlateConfig())

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        self.do_simulation(a, 1)
        obs = self.get_current_obs()
        reward, info = self._default_reward()
        terminated = bool(info.get("success", False))
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def reset_model(self, seed: int | None = None, options: dict | None = None):
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        self.set_state(self.init_qpos.copy(), np.zeros_like(self.init_qvel))
        if self.task is None:
            self.task = DishwasherPlateTask(self, DishwasherPlateConfig())
        self.task.reset(seed=seed, randomize=False)
        if len(self._arm_qpos_idx) > 0:
            self.q_des = self.data.qpos[self._arm_qpos_idx].copy()
        else:
            self.q_des = np.array([])
        if self.observation_space is None:
            obs = self.get_current_obs()
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)
        return self.get_current_obs()

    def get_current_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        plate_q = self.data.qpos[self._qadr_plate:self._qadr_plate+7]
        plate_d = self.data.qvel[self._dadr_plate:self._dadr_plate+6]
        obs_components: list[np.ndarray] = [qpos, qvel, plate_q, plate_d]
        if self.include_tcp:
            try:
                tcp_sid = self._sid_plate_center
                tcp_p = self.data.site_xpos[tcp_sid].copy()
                tcp_quat = np.empty(4, dtype=np.float64)
                mat9 = np.asarray(self.data.site_xmat[tcp_sid], dtype=np.float64)
                mujoco.mju_mat2Quat(tcp_quat, mat9)
                obs_components += [tcp_quat, tcp_p]
            except Exception:
                pass
        if self.include_rel:
            plate_pos = plate_q[3:6]
            target_pos = self.data.site_xpos[self._sid_target].copy()
            obs_components += [target_pos - plate_pos]
        return np.concatenate([c.ravel() for c in obs_components]).astype(np.float32)

    def _default_reward(self):
        return self.task.reward()

    def _freejoint_addr(self, body_id: int) -> tuple[int, int]:
        j0 = int(self.model.body_jntadr[body_id])
        jn = int(self.model.body_jntnum[body_id])
        for j in range(j0, j0 + jn):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                qadr = int(self.model.jnt_qposadr[j])
                dadr = int(self.model.jnt_dofadr[j])
                return qadr, dadr
        raise RuntimeError("Body has no free joint")

    def _find_arm_hinges(self, prefix: str, count: int) -> np.ndarray:
        idxs: list[int] = []
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if name.startswith(prefix) and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                idxs.append(int(self.model.jnt_qposadr[j]))
        return np.array(idxs[:count], dtype=np.int64)

    def _table_to_world(self, rel_xyz: np.ndarray) -> np.ndarray:
        try:
            tbid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table")
            return self.model.body_pos[tbid].copy() + np.asarray(rel_xyz, dtype=np.float64)
        except Exception:
            return np.asarray(rel_xyz, dtype=np.float64)