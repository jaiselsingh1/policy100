import time
import numpy as np
import gymnasium as gym
import mujoco
from typing import List, Dict, Any

import policy100.envs 
from controller import DiffIKController

Keypoint = Dict[str, Any]

def get_quat_from_euler(r: float, p: float, y: float) -> np.ndarray:
    """Creates a quaternion from Euler angles with 'xyz' sequence."""
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, np.array([r, p, y], dtype=np.float64), "xyz")
    return quat

def main() -> None:
    env = gym.make("XArmDishwasher-v0", render_mode="human")
    env.reset()
    
    unwrapped = env.unwrapped
    model = unwrapped.model
    data = unwrapped.data

    # DOF Extraction
    arm_joint_names = [f"joint{i}" for i in range(1, 8)]
    arm_dof_ids: List[int] = []
    
    for name in arm_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_adr = int(model.jnt_dofadr[jid])
        arm_dof_ids.append(dof_adr)

    # Initialize Controller
    controller = DiffIKController(
        model, 
        data, 
        site_name="tcp_site",
        dof_indices=arm_dof_ids 
    )

    # Define Geometry
    plate_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    
    plate_pos = data.site_xpos[plate_site_id].copy()

    # Define Initial Orientation
    quat_down = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat_down, data.site_xmat[tcp_id])

    keypoints: List[Keypoint] = [
        {
            "name": "HOVER_HIGH",
            "pos": plate_pos + np.array([0, 0, 0.15]), 
            "quat": quat_down, 
            "gripper": -1.0
        },
        {
            "name": "HOVER_LOW",
            "pos": plate_pos + np.array([0, 0, 0.05]), 
            "quat": quat_down,
            "gripper": -1.0
        },
        {
            "name": "GRASP_TARGET",
            "pos": plate_pos + np.array([0, 0, 0.015]), 
            "quat": quat_down,
            "gripper": -1.0
        }
    ]

    print("Starting Debug Loop...")

    for kp in keypoints:
        print(f"Executing: {kp['name']}")
        
        # Convergence loop
        for _ in range(400):
            delta_q = controller.get_action(
                target_pos=kp['pos'],
                target_quat=kp['quat']
            )
            
            current_arm_q = data.qpos[arm_dof_ids]
            desired_q = current_arm_q + delta_q
            action = np.concatenate([desired_q, [kp['gripper']]]).astype(np.float32)
            
            env.step(action)
            time.sleep(0.002)

        # Telemetry
        current_tcp = data.site_xpos[tcp_id]
        dist_error = np.linalg.norm(current_tcp - kp['pos'])
        
        print(f"  -> Finished {kp['name']}")
        print(f"     Target Z: {kp['pos'][2]:.4f}")
        print(f"     Actual Z: {current_tcp[2]:.4f}")
        print(f"     Error:    {dist_error:.4f} m")

        if kp['name'] == "GRASP_TARGET":
            input("\nPaused at GRASP_TARGET. Press Enter to finish...")

    env.close()

if __name__ == "__main__":
    main()























    # # Orientation A: Pointing Down (Grab current orientation)
    # quat_down = np.zeros(4, dtype=np.float64)
    # mujoco.mju_mat2Quat(quat_down, data.site_xmat[tcp_id])
    
    # # Orientation B: Pointing Forward (90 deg pitch) for the rack
    # quat_side = get_quat_from_euler(0.0, np.pi / 2, 0.0)

    # # 5. Define Trajectory
    # keypoints: List[Keypoint] = [
    #     # --- PHASE 1: PICK ---
        # {
        #     "name": "HOVER",
        #     "pos": plate_pos + np.array([0, 0, 0.20]), 
        #     "quat": quat_down,
        #     "gripper": -1.0
    #     },
    #     {
    #         "name": "GRASP_APPROACH",
    #         "pos": plate_pos + np.array([0, 0, 0.035]), # Tuned for collision
    #         "quat": quat_down,
    #         "gripper": -1.0
    #     },
    #     {
    #         "name": "CLOSE_GRIPPER",
    #         "pos": plate_pos + np.array([0, 0, 0.035]),
    #         "quat": quat_down,
    #         "gripper": 1.0
    #     },
    #     {
    #         "name": "LIFT",
    #         "pos": plate_pos + np.array([0, 0, 0.30]),
    #         "quat": quat_down,
    #         "gripper": 1.0
    #     },
    #     # --- PHASE 2: ORIENT ---
    #     {
    #         "name": "ROTATE_VERTICAL",
    #         "pos": np.array([0.45, 0.10, 0.35]),
    #         "quat": quat_side,
    #         "gripper": 1.0
    #     },
    #     # --- PHASE 3: INSERT ---
    #     {
    #         "name": "ALIGN_RACK",
    #         "pos": rack_pos + np.array([-0.15, 0, 0.1]), 
    #         "quat": quat_side,
    #         "gripper": 1.0
    #     },
    #     {
    #         "name": "LOWER_TO_SLOT",
    #         "pos": rack_pos + np.array([-0.15, 0, 0.02]), 
    #         "quat": quat_side,
    #         "gripper": 1.0
    #     },
    #     {
    #         "name": "INSERT_PUSH",
    #         "pos": rack_pos + np.array([0.0, 0, 0.02]), 
    #         "quat": quat_side,
    #         "gripper": 1.0
    #     },
    #     {
    #         "name": "RELEASE",
    #         "pos": rack_pos + np.array([0.0, 0, 0.02]), 
    #         "quat": quat_side,
    #         "gripper": -1.0
    #     }
    # ]

    # print("Starting execution loop...")

    # for kp in keypoints:
    #     print(f"Executing: {kp['name']}")
        
    #     # Run mini-loop to allow IK convergence
    #     for _ in range(300):
    #         # Calculate IK (Returns shape (7,))
    #         delta_q = controller.get_action(
    #             target_pos=kp['pos'],
    #             target_quat=kp['quat']
    #         )
            
    #         # Get current arm state
    #         current_arm_q = data.qpos[arm_dof_ids]
            
    #         # Apply Delta
    #         desired_q = current_arm_q + delta_q
            
    #         # Construct Action (7 arm joints + 1 gripper)
    #         action = np.concatenate([desired_q, [kp['gripper']]]).astype(np.float32)
            
    #         env.step(action)
    #         time.sleep(0.002)

    # print("completed loop")
    # time.sleep(2.0)
    # env.close()