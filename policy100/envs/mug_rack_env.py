import mujoco
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces
from pathlib import Path

from ..tasks.mug_rack import MugRackTask, MugRackConfig

class MugRackEnv(MujocoEnv):
    """
    MuJoCo environment for hanging a mug on a rack.

    This environment loads the `mug_rack_task.xml` asset, spawns a mug with a free joint on a table
    and provides a continuous control interface for a robot arm (if present) plus a single gripper
    command.  Observations include joint positions/velocities and the mug pose.  Rewards are
    computed by the associated `MugRackTask`.
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
        # Save parameters
        self.ctrl_scale = float(ctrl_scale)
        self.gr_ctrl_scale = float(gr_ctrl_scale)
        self.include_tcp = bool(include_tcp)
        self.include_rel = bool(include_rel)
        self.table_z = float(table_z)
        self.observation_space = None

        # Determine XML path relative to this file if not provided
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent / "assets" / "mug_rack_task.xml"
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

        # Cache body and site identifiers
        self._bid_mug = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mug")
        self._sid_handle = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "handle_hole")
        self._sid_hook = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hook_2_target")

        # Determine qpos and qvel addresses for the mug free joint
        self._qadr_mug, self._dadr_mug = self._freejoint_addr(self._bid_mug)

        # Attempt to find 7 hinge joints for the arm; if none exist the array will be empty
        if self.model.njnt > 0:
            try:
                self._arm_qpos_idx = self._find_arm_hinges(prefix="joint", count=7)
            except Exception:
                self._arm_qpos_idx = np.array([], dtype=np.int64)
        else:
            self._arm_qpos_idx = np.array([], dtype=np.int64)

        # Define action space: joint deltas (if any) plus a single gripper channel
        self.act_dim = len(self._arm_qpos_idx) + 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

        # Attach task
        self.task = MugRackTask(self, MugRackConfig())

    def step(self, action):
        # Apply the action and advance the simulator
        a = np.asarray(action, dtype=np.float64)
        self.do_simulation(a, 1)
        obs = self.get_current_obs()
        reward, info = self._default_reward()
        terminated = bool(info.get("success", False))
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def reset_model(self, seed: int | None = None, options: dict | None = None):
        # Reset state to the initial pose
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        self.set_state(self.init_qpos.copy(), np.zeros_like(self.init_qvel))

        # Reset the task (spawns objects)
        if self.task is None:
            self.task = MugRackTask(self, MugRackConfig())
        self.task.reset(seed=seed, randomize=False)

        # Sync desired joint positions
        if len(self._arm_qpos_idx) > 0:
            self.q_des = self.data.qpos[self._arm_qpos_idx].copy()
        else:
            self.q_des = np.array([])

        # Build observation space on first reset
        if self.observation_space is None:
            obs = self.get_current_obs()
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

        return self.get_current_obs()

    def get_current_obs(self) -> np.ndarray:
        # Basic observation: qpos and qvel for all degrees of freedom
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        mug_q = self.data.qpos[self._qadr_mug:self._qadr_mug+7]
        mug_d = self.data.qvel[self._dadr_mug:self._dadr_mug+6]

        obs_components: list[np.ndarray] = [qpos, qvel, mug_q, mug_d]

        if self.include_tcp:
            # Provide the mug orientation and position as orientation + position
            try:
                tcp_sid = self._sid_handle
                tcp_p = self.data.site_xpos[tcp_sid].copy()
                tcp_quat = np.empty(4, dtype=np.float64)
                mat9 = np.asarray(self.data.site_xmat[tcp_sid], dtype=np.float64)
                mujoco.mju_mat2Quat(tcp_quat, mat9)
                obs_components += [tcp_quat, tcp_p]
            except Exception:
                pass

        if self.include_rel:
            # Relative vector from handle to hook
            handle_pos = self.data.site_xpos[self._sid_handle].copy()
            hook_pos = self.data.site_xpos[self._sid_hook].copy()
            obs_components += [hook_pos - handle_pos]

        return np.concatenate([c.ravel() for c in obs_components]).astype(np.float32)

    def _default_reward(self):
        return self.task.reward()

    # Helper: find free joint qpos and qvel address for a body
    def _freejoint_addr(self, body_id: int) -> tuple[int, int]:
        j0 = int(self.model.body_jntadr[body_id])
        jn = int(self.model.body_jntnum[body_id])
        for j in range(j0, j0 + jn):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                qadr = int(self.model.jnt_qposadr[j])
                dadr = int(self.model.jnt_dofadr[j])
                return qadr, dadr
        raise RuntimeError("Body has no free joint")

    # Helper: find hinge joints with a given prefix
    def _find_arm_hinges(self, prefix: str, count: int) -> np.ndarray:
        idxs: list[int] = []
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if name.startswith(prefix) and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                idxs.append(int(self.model.jnt_qposadr[j]))
        return np.array(idxs[:count], dtype=np.int64)

    def _table_to_world(self, rel_xyz: np.ndarray) -> np.ndarray:
        """
        Convert a position relative to the table (z=0 at the tabletop) to world coordinates.  If the
        table body exists, add its world position; otherwise return the input.
        """
        try:
            tbid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table")
            return self.model.body_pos[tbid].copy() + np.asarray(rel_xyz, dtype=np.float64)
        except Exception:
            return np.asarray(rel_xyz, dtype=np.float64)