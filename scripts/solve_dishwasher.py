import time
import numpy as np
import gymnasium as gym
import mujoco
import policy100.envs
# import controller
from controller import DiffIKController, IKConfig, make_quaternion

def main():
    # setup environment
    env = gym.make("XArmDishwasher-v0", render_mode="human")
    obs, info = env.reset()
    
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # setup Controller with precision config
    ik_config = IKConfig(
        damping=1e-3,
        max_delta_q=0.05, 
        pos_gain=1.0,
        ori_gain=1.0,
        tolerance_pos=0.002,  # 2mm tolerance for grasping
        tolerance_ori=0.05
    )
    
    ik = DiffIKController(
        model, data, 
        site_name="tcp_site", 
        config=ik_config
    )

    # targets
    plate_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
    plate_pos = data.site_xpos[plate_sid].copy()
    
    # orientation -> gripper pointing DOWN (pitch=180 deg)
    quat_down = make_quaternion(roll=0, pitch=np.pi, yaw=0)

    # Structure: (Position, Gripper_Value, Label)
    # Gripper: 0.85 = Open, 0.0 = Closed
    trajectory = [
        {
            "name": "HOVER_APPROACH",
            "pos": plate_pos + np.array([0.0, 0.0, 0.20]), # 20cm above
            "quat": quat_down,
            "gripper": 0.85 
        },
        {
            "name": "DESCEND_TO_GRASP",
            "pos": plate_pos + np.array([0.0, 0.0, 0.025]), # 2.5cm offset for fingers
            "quat": quat_down,
            "gripper": 0.85
        },
        {
            "name": "CLOSE_GRIPPER",
            "pos": plate_pos + np.array([0.0, 0.0, 0.025]),
            "quat": quat_down,
            "gripper": 0.0 
        }
    ]

    print("\n--- Starting Minimal Grasp Test ---")
    
    # 5. Execution Loop
    for point in trajectory:
        target_pos = point["pos"]
        target_quat = point["quat"]
        target_gripper = point["gripper"]
        
        print(f"Moving to: {point['name']}...")
        
        # IK Convergence Loop
        for i in range(500):
            # A. Compute IK
            new_q, pos_err, ori_err = ik.step_toward(target_pos, target_quat)
            
            # B. Apply to Simulation (Kinematic Override)
            # We write directly to qpos to isolate geometric errors from control errors
            data.qpos[ik.qpos_indices] = new_q
            data.qvel[ik.dof_indices] = 0.0 # Stop movement
            
            # Apply Gripper Action (last actuator)
            # Find gripper actuator index (usually last)
            data.ctrl[-1] = target_gripper
            
            # C. Step Physics to update sites/sensors
            mujoco.mj_forward(model, data)
            
            # D. Render
            env.render()
            time.sleep(0.005)
            
            # E. Check Convergence
            if ik.is_converged(target_pos, target_quat):
                print(f" -> Reached {point['name']} (Err: {pos_err:.4f}m)")
                break
        
        # optional 
        if point["name"] == "DESCEND_TO_GRASP":
            print("   [Paused] Check alignment in viewer. Press Enter to grasp...")
            # input() # Uncomment to manually step through
            time.sleep(1.0) # Short pause for visual check

    print("--- Grasp Sequence Complete ---")
    
    # hold the final pose
    while True:
        env.render()
        time.sleep(0.1)

if __name__ == "__main__":
    main()