import time 
import numpy as np 
import gymnasium as gym 
import mujoco 
from controller import DiffIKController 
from policy100.envs import xarm_env

env = gym.make("XArmDishwasher-v0", render_mode="human")
obs, info = env.reset()

model = env.unwrapped.model 
data = env.unwrapped.data 


# find the 7 arm DOFs
arm_joint_names = [f"joint{i}" for i in range(1, 8)]
arm_qpos_idx = []
for name in arm_joint_names:
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j_id == -1:
        raise RuntimeError(f"Joint{name} not found in model")
    arm_qpos_idx.append(int(model.jnt_qposadr[j_id]))
arm_qpos_idx = np.array(arm_qpos_idx, dtype=np.int64)

ik = DiffIKController(
    model = model, 
    data = data, 
    site_name = "tcp_site", 
    dof_indices = arm_qpos_idx, 
    damping = 1e-4, 
    step_size = 0.5
)

plate_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
rack_sid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_slot")

if plate_sid == -1 or rack_sid == -1:
    raise RuntimeError("plate_center or target_slot site not found")

plate_pos = data.site_xpos[plate_sid].copy()
rack_pos  = data.site_xpos[rack_sid].copy()

def quat_from_euler(roll, pitch, yaw):
    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), 'xyz')
    return q

# these need to be tweaked
quat_lean     = quat_from_euler(0.0, 0.0, 0.0)       # start with neutral
quat_vertical = quat_from_euler(0.0, np.pi / 2, 0.0) # plate upright for insertion

keypoints = [
    {
        "name": "HOVER_ABOVE_PLATE",
        "pos": plate_pos + np.array([0.0, 0.0, 0.15]),
        "quat": quat_lean,
        "gripper": -1.0,  # open
    },
    {
        "name": "DESCEND_TO_RIM",
        "pos": plate_pos + np.array([0.0, 0.0, 0.03]),
        "quat": quat_lean,
        "gripper": -1.0,
    }
]   

def go_to_keypoint(kp, iters=300, dt=0.002):
    """Move arm using IK toward kp['pos'], kp['quat'], controlling gripper via action."""
    target_pos = kp["pos"]
    target_quat = kp["quat"]
    grip_cmd = kp["gripper"]

    for _ in range(iters):
        # Differential joint update from IK
        d_q = ik.get_action(target_pos, target_quat)  # shape (7,)
        q = data.qpos[arm_qpos_idx]
        new_q = q + d_q

        # Write new joint positions directly
        data.qpos[arm_qpos_idx] = new_q

        # Build action: 7 dummy values (unused) + gripper command
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[-1] = grip_cmd

        env.step(action)
        time.sleep(dt)

    # Simple debug: position error at the end of this segment
    tcp_pos = data.site_xpos[ik.site_id]
    err = np.linalg.norm(tcp_pos - target_pos)
    print(f"{kp['name']} | final pos error: {err:.4f}")


# Run the full scripted trajectory
for kp in keypoints:
    print(f"--- Executing {kp['name']} ---")
    go_to_keypoint(kp, iters=300)