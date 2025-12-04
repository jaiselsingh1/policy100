from gymnasium.envs.registration import register
from .xarm_env import XArmEnv

# Register the Mug Rack task
register(
    id="XArmMugRack-v0",
    entry_point="policy100.envs.xarm_env:XArmEnv",
    max_episode_steps=1000,
    kwargs={"task_name": "mug_rack"}
)

# Register the Dishwasher task
register(
    id="XArmDishwasher-v0",
    entry_point="policy100.envs.xarm_env:XArmEnv",
    max_episode_steps=1000,
    kwargs={"task_name": "dishwasher"}
)

__all__ = ["XArmEnv"]