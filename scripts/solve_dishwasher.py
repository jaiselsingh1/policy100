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
    step_size = 0.05
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

def go_to_keypoint(kp, iters=300, dt=0.01, pos_tol=1e-3):
    """
    Move arm using IK toward kp['pos'], kp['quat'], controlling gripper via action.
    Uses small, clamped joint updates and stops early when close enough.
    """
    target_pos = kp["pos"]
    target_quat = kp["quat"]
    grip_cmd = kp["gripper"]
    max_step = ik.step_size

    for i in range(iters):
        # Current TCP position
        tcp_pos = data.site_xpos[ik.site_id].copy()
        pos_err = np.linalg.norm(tcp_pos - target_pos)

        # Early stop if we're close enough
        if pos_err < pos_tol:
            print(f"{kp['name']} | converged at iter {i}, pos_err={pos_err:.4f}")
            break

        # Differential joint update from IK
        d_q = ik.get_action(target_pos, target_quat)  # shape (7,)

        # Clamp the update magnitude for smooth motion
        norm = np.linalg.norm(d_q)
        if norm > max_step and norm > 0:
            d_q *= max_step / norm

        # Apply kinematic update
        q = data.qpos[arm_qpos_idx]
        new_q = q + d_q
        data.qpos[arm_qpos_idx] = new_q

        # Zero arm velocities to avoid weird dynamics
        data.qvel[arm_qpos_idx] = 0.0

        # Build action: only gripper is used
        # action = np.zeros(env.action_space.shape, dtype=np.float32)
        # action[-1] = grip_cmd

        # env.step(action)
        time.sleep(dt)

    # Final error report
    tcp_pos = data.site_xpos[ik.site_id]
    err = np.linalg.norm(tcp_pos - target_pos)
    print(f"{kp['name']} | final pos error: {err:.4f}")



# Run the full scripted trajectory
for kp in keypoints:
    print(f"{kp['name']}")
    go_to_keypoint(kp, iters=300)