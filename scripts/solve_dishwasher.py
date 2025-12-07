"""
solve_dishwasher.py - Grasp Reliability Validation

Methodology: Independent Bernoulli Trials.
1. Reset Environment (Clean State)
2. Approach -> Descend -> Grasp -> Lift
3. Validate Success (Is plate height > threshold?)
4. Repeat
"""

import time
import numpy as np
import gymnasium as gym
import mujoco

# 1. Trigger environment registration
import policy100.envs 

from controller import DiffIKController, IKConfig, make_quaternion

def run_trial(env, trial_idx: int) -> bool:
    """
    Executes a single grasp attempt from a fresh reset.
    Returns: True if grasp was successful (plate lifted).
    """
    # A. Hard Reset: Ensure independence between trials
    obs, info = env.reset()
    
    # B. Re-initialize Data/Model pointers after reset
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # C. Setup Controller (High precision for grasping)
    ik_config = IKConfig(
        damping=1e-3,
        max_delta_q=0.05, 
        pos_gain=1.0, 
        ori_gain=1.0,
        tolerance_pos=0.002, 
        tolerance_ori=0.05
    )
    
    ik = DiffIKController(model, data, site_name="tcp_site", config=ik_config)

    # D. Perception: Find the plate *in this specific episode*
    plate_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
    plate_pos = data.site_xpos[plate_sid].copy()
    
    # Define Target Orientation (Gripper Down)
    quat_down = make_quaternion(roll=0, pitch=np.pi, yaw=0)

    # E. Define Trajectory: Hover -> Grasp -> Lift (Validation)
    trajectory = [
        {
            "name": "HOVER",
            "pos": plate_pos + np.array([0.0, 0.0, 0.20]), 
            "quat": quat_down, "gripper": 0.85 
        },
        {
            "name": "DESCEND",
            "pos": plate_pos + np.array([0.0, 0.0, 0.025]), 
            "quat": quat_down, "gripper": 0.85 
        },
        {
            "name": "CLOSE",
            "pos": plate_pos + np.array([0.0, 0.0, 0.025]),
            "quat": quat_down, "gripper": 0.0  # Close
        },
        {
            "name": "LIFT_VALIDATION",
            "pos": plate_pos + np.array([0.0, 0.0, 0.20]), 
            "quat": quat_down, "gripper": 0.0  # Keep Closed
        }
    ]

    print(f"\n--- Trial {trial_idx + 1} ---")
    
    for point in trajectory:
        target_pos = point["pos"]
        target_quat = point["quat"]
        
        # IK Execution Loop
        for _ in range(500):
            # 1. Compute
            new_q, pos_err, _ = ik.step_toward(target_pos, target_quat)
            
            # 2. Act (Kinematic Override)
            data.qpos[ik.qpos_indices] = new_q
            data.qvel[ik.dof_indices] = 0.0
            if len(data.ctrl) > 0: data.ctrl[-1] = point["gripper"]
            
            # 3. Step Physics
            mujoco.mj_forward(model, data)
            env.render()
            time.sleep(0.002)
            
            # 4. Convergence
            if ik.is_converged(target_pos, target_quat):
                break
        
        # Brief pause to verify visual state
        if point["name"] == "CLOSE":
            time.sleep(0.2)

    # F. Validation: Did the plate actually move up?
    # We check the current Z height of the plate site
    current_plate_z = data.site_xpos[plate_sid][2]
    # Initial plate Z is usually ~0.0 (on table) or ~0.02 (thickness)
    # If we lifted to +0.20, it should be well above 0.10
    success = current_plate_z > 0.10
    
    if success:
        print(f"Result: [PASS] Plate Z = {current_plate_z:.3f}m")
    else:
        print(f"Result: [FAIL] Plate Z = {current_plate_z:.3f}m (Grasp Slipped)")
        
    return success

def main():
    env = gym.make("XArmDishwasher-v0", render_mode="human")
    
    n_trials = 3
    successes = 0
    
    for i in range(n_trials):
        if run_trial(env, i):
            successes += 1
        time.sleep(1.0) # Pause between trials
            
    print(f"\nReliability Score: {successes}/{n_trials} ({successes/n_trials*100:.1f}%)")
    
    # Hold final state
    while True:
        env.render()
        time.sleep(0.1)

if __name__ == "__main__":
    main()