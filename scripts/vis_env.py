import argparse
from dataclasses import dataclass
from typing import Literal
import time
import numpy as np
import tyro 

from policy100.envs import MugRackEnv, DishwasherPlateEnv

@dataclass
class VisConfig:
    env: Literal["mug", "dish"] = "dish"
    steps: int = 1000


def main() -> None:
    args = tyro.cli(VisConfig)

    if args.env == "mug":
        env = MugRackEnv(render_mode="human")
    else:
        env = DishwasherPlateEnv(render_mode="human")

    obs, info = env.reset()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    
    for i in range(args.steps):
        env.step(action)
        if env.render_mode == "human":
            env.render()
            # Sleep a bit to control frame rate
            time.sleep(0.002)


if __name__ == "__main__":
    main()