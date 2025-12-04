# Policy100 Project

This repository contains a lightweight implementation for two robotic manipulation tasks inspired by the original GPS code base.  The goal is to provide a minimal starting point for developing and training policies for:

- **Hang a mug on a rack** – pick up a mug from the table and hang it on the second hook of a rack.
- **Place a plate into a dishwasher rack** – pick up a plate from the table and insert it into a target slot in the rack.

The repository is organised to be easy to understand and extend.  It does **not** depend on JAX or DeepMind’s `mujoco-warp` bindings.  All simulations use [`gymnasium`](https://gymnasium.farama.org/) with MuJoCo as the physics engine.

## Structure

```text
policy100/
├── assets/                # MuJoCo XML files defining the tasks
│   ├── mug_rack_task.xml
│   └── plate_dishwasher_task.xml
├── envs/                  # Environment classes for each task
│   ├── mug_rack_env.py
│   └── dishwasher_plate_env.py
├── tasks/                 # Task definitions (reward functions, metrics, etc.)
│   ├── mug_rack.py
│   └── dishwasher_plate.py
├── scripts/               # Example scripts (visualisation, training)
│   └── vis_env.py
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

### Installation

1. Create a Python 3.11 virtual environment (or use a conda environment).
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the visualisation script to make sure everything loads:

```bash
python -m policy100.scripts.vis_env --env mug
```

The script will create an instance of the selected environment and render it using `gymnasium`.

## Environments

Two environments are provided as simple wrappers around `gymnasium`’s `MujocoEnv`:

- **`MugRackEnv`** loads `assets/mug_rack_task.xml` and returns observations consisting of joint positions, velocities and key object poses.  The agent’s action is a vector of joint control inputs for the arm plus a gripper command.  The reward is defined in `tasks/mug_rack.py` and encourages lifting the mug, moving it to the hook and hanging it successfully.

- **`DishwasherPlateEnv`** loads `assets/plate_dishwasher_task.xml` and returns a similar observation vector.  The reward in `tasks/dishwasher_plate.py` encourages lifting the plate, moving it over the rack and inserting it into the target slot.

Both environments are intentionally minimal: they do not randomise object positions by default, do not require any domain randomisation and do not use vectorised environments.  They are meant as a starting point for quick experiments.

## Next Steps

The high‑level roadmap for this project is summarised below (based on the to‑do list in the provided screenshot):

- **Parameter space definition:** sample initial positions of the mug and plate within ±10 cm of the default pose and a single yaw rotation.  The rack positions are fixed.
- **Generate demonstrations:** use the keyframe definitions in the XML files to generate a small number of demonstrations.  You can interpolate between keyframes to produce trajectories and warp the timing to study how speed affects success.
- **Train a policy:** collect 200–300 demonstrations and train a control policy using a reinforcement learning library such as Stable Baselines3.  Evaluate failures by visual inspection and plan recovery behaviours.
- **Performance comparison:** measure the simulation frame rate using MuJoCo alone, varying the number of environments per GPU.

These steps are not fully automated in this repository but the structure has been designed to make it easy to add scripts for demonstration collection and policy training.