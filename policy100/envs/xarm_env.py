import os
import mujoco
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces
from pathlib import Path
from typing import Literal, Any, Type

# import tasks 
from policy100.tasks.mug_rack import MugRackTask, MugRackConfig
from policy100.tasks.dishwasher_plate import DishwasherPlateTask, DishwasherPlateConfig

@dataclass(frozen=True)
class TaskRegistryEntry:
    "defines a static config for the task aka dishwasher, etc"
    xml_file: str 
    task_class: Type[Any] # class itself (eg. MugRackTask)
    task_config: Type[Any] # config class
    obj_body_name: str 
    tcp_site_name: str
    target_site_name: str 

# mapping task names to their configs 
TASK_CONFIGS = {
    "mug_rack": TaskRegistryEntry(
        xml_file="mug_rack_task.xml",
        task_class=MugRackTask,
        task_config=MugRackConfig,
        obj_body_name="mug",
        tcp_site_name="handle_hole",
        target_site_name="hook_2_target",
    ),
    "dishwasher": TaskRegistryEntry(
       #xml_file="plate_dishwasher_task.xml",
        xml_file="plate_dishwasher_mjcf.xml", 
        task_class=DishwasherPlateTask,
        task_config=DishwasherPlateConfig,
        obj_body_name="plate",
        tcp_site_name="plate_center",
        target_site_name="target_slot",
    ),
}


class XArmEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self,
                    task_name: Literal["mug_rack", "dishwasher"] = "dishwasher",
                    model_path: str | None = None,
                    frame_skip: int = 5,
                    include_tcp: bool = True,
                    include_rel: bool = True,
                    **kwargs):
        if task_name not in TASK_CONFIGS:
            raise ValueError(f"Task '{task_name}' not found. Options: {list(TASK_CONFIGS.keys())}")
        
        self.entry = TASK_CONFIGS[task_name]
        self.include_tcp = include_tcp # adding tcp into the obseration vector or not 
        self.include_rel = include_rel # target position - current obj position 

        if model_path is None:
            root_dir = Path(__file__).resolve().parent.parent
            
            path_xarm = root_dir / "assets" / "xarm" / self.entry.xml_file
            if path_xarm.exists():
                model_path = str(path_xarm)
            else:
                raise ValueError("model_path is not defined")

        # We initialize with a dummy obs space first, then correct it later
        MujocoEnv.__init__(self, model_path, frame_skip, observation_space=None, **kwargs)

        # cache the body ids 
        self._bid_obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.entry.obj_body_name)
        self._sid_tcp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.entry.tcp_site_name)
        self._sid_target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.entry.target_site_name)

        # check that the ids are found 
        if self._bid_obj == -1:
            raise ValueError(f"Body '{self.entry.obj_body_name}' not found in XML.")
        if self._sid_tcp == -1:
            raise ValueError(f"Site '{self.entry.tcp_site_name}' not found in XML.")
        if self._sid_target == -1:
            raise ValueError(f"Site '{self.entry.target_site_name}' not found in XML.")
        
        # free joint addresses for the object
        self._qadr_obj, self._dadr_obj = self._freejoint_addr(self._bid_obj)
        # auto-detect xArm joints (looks for joints starting with "joint")
        # expect 7 arm joints.
        if self.model.njnt > 0:
            self._arm_qpos_idx = self._find_arm_hinges(prefix="joint", count=7)
        else:
            self._arm_qpos_idx = np.array([], dtype=np.int64)

        # 7 arm joints + 1 gripper actuator
        self.act_dim = len(self._arm_qpos_idx) + 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

        # Set observation space now that we can measure it
        obs = self.get_current_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

        # initialize the task logic 
        TaskClass = self.entry.task_class
        ConfigClass = self.entry.task_config # Fixed typo here (was self.entry.conf)
        self.task = TaskClass(self, ConfigClass())

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        self.do_simulation(a, self.frame_skip)
        
        # Render check for human mode
        if self.render_mode == "human":
            self.render()
            
        obs = self.get_current_obs()

        # reward/success is taken care of by the task 
        reward, info = self.task.reward()

        terminated = bool(info.get("success", False)) # Fixed typo "sucess"
        truncated = False

        return obs, float(reward), terminated, truncated, info 
    
    def reset_model(self):
        # reset physics to initial XML state
        self.set_state(self.init_qpos, self.init_qvel)

        # reset Task logic (randomization, spawning, etc.)
        self.task.reset(randomize=False)

        # sync desired position for controllers (prevents arm snapping on reset)
        if len(self._arm_qpos_idx) > 0:
            # logic for controller reset if needed
            pass

        return self.get_current_obs()
    
    def get_current_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # Generic access using the cached addresses from __init__
        # Free joint has 7 qpos (x,y,z, quat) and 6 qvel (vx,vy,vz, wx,wy,wz)
        obj_q = self.data.qpos[self._qadr_obj:self._qadr_obj+7]
        obj_d = self.data.qvel[self._dadr_obj:self._dadr_obj+6]

        obs_components = [qpos, qvel, obj_q, obj_d]

        if self.include_tcp:
            tcp_p = self.data.site_xpos[self._sid_tcp].copy()
            tcp_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(tcp_quat, self.data.site_xmat[self._sid_tcp])
            obs_components += [tcp_quat, tcp_p]

        if self.include_rel:
            target_p = self.data.site_xpos[self._sid_target].copy()
            tcp_p = self.data.site_xpos[self._sid_tcp].copy()
            obs_components.append(target_p - tcp_p)

        return np.concatenate([c.ravel() for c in obs_components]).astype(np.float32)
    
    # helper methods 
    def _freejoint_addr(self, body_id: int) -> tuple[int, int]:
        """Finds the qpos and qvel addresses for the free joint of a given body."""
        j0 = int(self.model.body_jntadr[body_id])
        jn = int(self.model.body_jntnum[body_id])
        for j in range(j0, j0 + jn):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j])
        raise RuntimeError(f"Body {body_id} has no free joint")

    def _find_arm_hinges(self, prefix: str, count: int) -> np.ndarray:
        """Finds 'count' number of hinge joints starting with 'prefix'."""
        idxs = []
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if name.startswith(prefix) and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                idxs.append(int(self.model.jnt_qposadr[j]))
        return np.array(idxs[:count], dtype=np.int64)

    def _table_to_world(self, rel_xyz: np.ndarray) -> np.ndarray:
        """
        Convert a position relative to the table (z=0 at the tabletop) to world coordinates.
        Useful if tasks need to spawn objects relative to the table.
        """
        try:
            tbid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table")
            if tbid != -1:
                return self.model.body_pos[tbid].copy() + np.asarray(rel_xyz, dtype=np.float64)
            return np.asarray(rel_xyz, dtype=np.float64)
        except Exception:
            return np.asarray(rel_xyz, dtype=np.float64)