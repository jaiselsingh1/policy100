import argparse
import time
import numpy as np
import tyro 
import gymnasium as gym
from dataclasses import dataclass
from typing import Literal

# import your package to trigger registration
import policy100.envs 

@dataclass
class VisConfig:
    env: Literal["mug", "dish"] = "dish"
    steps: int = 1000


def main() -> None:
    args = tyro.cli(VisConfig)

    # map short name to registered Gym ID
    env_id = "XArmMugRack-v0" if args.env == "mug" else "XArmDishwasher-v0"
    
    env = gym.make(env_id, render_mode="human")
    obs, info = env.reset()
    
    action = np.zeros(env.action_space.shape, dtype=np.float32)    
    print(action)
    for i in range(args.steps):
        env.step(action)
        # now let's reset the env as well
        # gym.make loop usually handles rendering, but sleep helps visualization speed
        time.sleep(0.002)


if __name__ == "__main__":
    main()