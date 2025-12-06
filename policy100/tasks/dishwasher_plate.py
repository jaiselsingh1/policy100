import mujoco 
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

@dataclass(frozen=True)
class DishwasherPlateConfig:
    name: str = "dishwasher_plate"
    # success criteria
    tol_xy: float = 0.05        # Success tolerance (meters)
    lift_height: float = 0.02   # Height above table to count as "lifted"
    
    # Reward shaping
    reach_alpha: float = 4.0    # Exponential scaling for distance penalty
    lift_bonus: float = 0.25    # Bonus for lifting the plate
    
    # Randomization - optional for reset 
    randomize: bool = False


class DishwasherPlateTask:
    def __init__(self, env, config: Optional[DishwasherPlateConfig] = None):
        self.env = env 
        self.cfg = config or DishwasherPlateConfig()

        self._sid_plate = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        self._sid_target = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "target_slot")

        self._qadr_plate = env._qadr_obj
        self._dadr_plate = env._dadr_obj

    def reset(self, seed: Optional[int] = None, randomize: bool = False):
        if seed is not None:
            np.random.seed(seed)

        # Copy the initial pose from the XML (stored in env.init_qpos)
        default_pose = self.env.init_qpos[self._qadr_plate : self._qadr_plate + 7]
        self.env.data.qpos[self._qadr_plate : self._qadr_plate + 7] = default_pose

        # Zero velocities and forward the model
        self.env.data.qvel[self._dadr_plate : self._dadr_plate + 6] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)


    def reward(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates the reward based on the state of the plate and target.
        """
        plate_pos = self.env.data.site_xpos[self._sid_plate]
        target_pos = self.env.data.site_xpos[self._sid_target]
        
        # Get Z height of the plate (index 2 in position vector is Z)
        # Note: qpos has 7 elements (3 pos, 4 quat). The Z pos is at index +2.
        plate_z = self.env.data.qpos[self._qadr_plate + 2]
        
        # Distance from plate center to target slot
        distance = np.linalg.norm(plate_pos - target_pos)
        
        # check if lifted (Z > table_height + threshold)
        # We assume env has a table_z attribute, or we default to 0.0 if not
        table_height = getattr(self.env, "table_z", 0.0)
        is_lifted = plate_z > (table_height + self.cfg.lift_height)
        
        # Shaped reward: Higher when closer to target
        reward = np.exp(-self.cfg.reach_alpha * distance)
        
        # Add discrete bonus for lifting
        if is_lifted:
            reward += self.cfg.lift_bonus
            
        # Success if close enough AND lifted (presumably into the rack)
        success = (distance < self.cfg.tol_xy) and is_lifted
        
        info = {
            "success": success,
            "distance": distance,
            "is_lifted": is_lifted
        }
        
        return float(reward), info





        

