import time
import numpy as np
import gymnasium as gym
import mujoco
import policy100.envs
import tqdm
from scipy.spatial.transform import Rotation as R

from controller import DiffIKController, IKConfig


def approach_dir_to_quat(approach_dir: np.ndarray, gripper_y_hint: np.ndarray = None) -> np.ndarray:
    "convert an approach direction into a quarternion for the gripper"

    gripper_z = approach_dir / np.linalg.norm(approach_dir)

    if gripper_y_hint is None:
        gripper_y_hint = np.array([0.0, 0.0, 1.0]) 

    # Gram-Schmidt: remove component parallel to Z
    gripper_y = gripper_y_hint - np.dot(gripper_y_hint, gripper_z) * gripper_z

    # Handle edge case: hint was parallel to Z
    if np.linalg.norm(gripper_y) < 1e-6:
        # Pick arbitrary perpendicular direction
        gripper_y = np.array([1.0, 0.0, 0.0])
        gripper_y = gripper_y - np.dot(gripper_y, gripper_z) * gripper_z

    gripper_y = gripper_y / np.linalg.norm(gripper_y)
    
    # Complete right-handed frame
    gripper_x = np.cross(gripper_y, gripper_z)
    gripper_x = gripper_x / np.linalg.norm(gripper_x)
    
    # Build rotation matrix: columns are the gripper axes in world frame
    R = np.column_stack([gripper_x, gripper_y, gripper_z])
    
    # Convert to quaternion
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    
    return quat

def get_grasp_pose_from_plate(model, data, plate_site_name: str = "plate_center"):
    """
    Compute grasp position and orientation from the plate's pose.
    
    Returns:
        (grasp_pos, grasp_quat, approach_dir)
    """
    plate_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, plate_site_name)
    if plate_sid == -1:
        raise RuntimeError(f"Site '{plate_site_name}' not found")
    
    mujoco.mj_forward(model, data)
    
    # Get plate pose
    plate_pos = data.site_xpos[plate_sid].copy()
    plate_mat = data.site_xmat[plate_sid].reshape(3, 3)
    
    # Plate Z-axis is the normal to the plate surface
    plate_normal = plate_mat[:, 2]
    plate_y = plate_mat[:, 1]
    
    # Gripper approaches direction that is perpindicular to the plate
    approach_dir = plate_y
    
    grasp_quat = approach_dir_to_quat(approach_dir, gripper_y_hint=plate_normal)
    
    return plate_pos, grasp_quat, approach_dir

def parametrized_grasp_pose(model, data, plate_site_name: str = "plate_center", offset_angle: float = 0):
    plate_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, plate_site_name)
    if plate_sid == -1:
        raise RuntimeError(f"Site '{plate_site_name}' not found")
    
    mujoco.mj_forward(model, data)

    # Get plate pose
    plate_pos = data.site_xpos[plate_sid].copy()
    plate_mat = data.site_xmat[plate_sid].reshape(3, 3)

    r = R.from_euler("z", [offset_angle])
    plate_rot = R.from_matrix(plate_mat) # represent the matrix as a rotation 
    # compose 
    composed_rot = r * plate_rot
    composed_mat = composed_rot.as_matrix()[0]

    approach_dir = composed_mat[:, 1] 
    grasp_quat = approach_dir_to_quat(approach_dir, gripper_y_hint=composed_mat[:, 2])
    
    return plate_pos, grasp_quat, approach_dir



def step_ik(env, ik, target_pos, target_quat, steps):
    model = env.unwrapped.model
    data = env.unwrapped.data

    for _ in tqdm.tqdm(range(steps)):
        dq = ik.compute(target_pos, target_quat=target_quat, use_nullspace=False)
        q = ik.get_current_joints()
        new_q = q + dq
        new_q = np.clip(new_q, ik.joint_limits[:, 0], ik.joint_limits[:, 1])

        data.ctrl[:7] = new_q
        mujoco.mj_step(model, data)
        env.render()


def set_gripper(env, value, steps):
    model = env.unwrapped.model
    data = env.unwrapped.data

    for _ in range(steps):
        data.ctrl[7] = value
        mujoco.mj_step(model, data)
        env.render()


def main():
    env = gym.make("XArmDishwasher-v0", render_mode="human")
    env.reset()

    model = env.unwrapped.model
    data = env.unwrapped.data

    ik = DiffIKController(
        model,
        data,
        site_name="link_tcp",
        config=IKConfig(
            damping=1e-3,
            max_delta_q=1.0,
            pos_gain=1.0,
            ori_gain=1.0,
            nullspace_gain=0.0,
            tolerance_pos=0.003,
            tolerance_ori=0.1,
        ),
    )

    # Get grasp pose from the plate's quaternian
    plate_pos, grasp_quat, approach_dir = parametrized_grasp_pose(model, data, offset_angle=-3*np.pi/4)
    # plate_pos, grasp_quat, approach_dir = get_grasp_pose_from_plate(model, data)
    
    print(f"Plate position: {plate_pos}")
    print(f"Approach direction: {approach_dir}")
    print(f"Grasp quaternion: {grasp_quat}")

    # Compute waypoints along the approach direction
    # hover -- offset along the approach direction
    hover_offset = 0.20  
    hover_pos = plate_pos - approach_dir * hover_offset  # minus because approach points toward plate
    
    # Grasp: small offset from plate center
    grasp_offset = -0.01
    grasp_pos = plate_pos - approach_dir * grasp_offset
    
    # Lift: for now, just go back to hover
    # lift_pos = hover_pos
    lift_pos = np.array([0.50+0.02, 0.22, 0.35])
    lift_quat = np.array([0, -0.707, 0.707, 0])

    drop_pos = np.array([0.50+0.02, 0.22, 0.15])

    print(f"\nWaypoints:")
    print(f"  Hover: {hover_pos}")
    print(f"  Grasp: {grasp_pos}")

    for _ in range(10):
        env.render()

    # Execute
    print("\nMoving to hover")
    step_ik(env, ik, hover_pos, grasp_quat, steps=700)

    print("Descending to grasp")
    step_ik(env, ik, grasp_pos, grasp_quat, steps=200)

    print("Closing gripper")
    set_gripper(env, value=255.0, steps=50)

    print("Lifting")
    step_ik(env, ik, lift_pos, lift_quat, steps=400)

    print("Lowering to rack")
    step_ik(env, ik, drop_pos, lift_quat, steps=200)

    print("open gripper")
    set_gripper(env, value=0.0, steps=50)

    print("Lifting adter drop")
    step_ik(env, ik, lift_pos, lift_quat, steps=250)

    # Check result
    mujoco.mj_forward(model, data)
    plate_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
    new_plate_pos = data.site_xpos[plate_sid].copy()
    
    print(f"\nPlate moved: {plate_pos} -> {new_plate_pos}")
    
    # Simple check: did the plate move significantly?
    displacement = np.linalg.norm(new_plate_pos - plate_pos)
    if displacement > 0.05:
        print(f"SUCCESS: Plate displaced by {displacement:.3f}m")
    else:
        print(f"FAILED: Plate barely moved ({displacement:.3f}m)")

    time.sleep(2.0)
    env.close()


if __name__ == "__main__":
    main()
