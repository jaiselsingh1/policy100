import numpy as np 
import gymnasium as gym 
import mujoco 
import policy100.envs 

"""
Debug: Check if the quaternion actually produces the right orientation
"""

import numpy as np
import mujoco


def approach_dir_to_quat(approach_dir: np.ndarray, gripper_y_hint: np.ndarray = None) -> np.ndarray:
    """Same function - let's trace through it"""
    
    print(f"\n=== approach_dir_to_quat debug ===")
    print(f"Input approach_dir: {approach_dir}")
    
    gripper_z = approach_dir / np.linalg.norm(approach_dir)
    print(f"gripper_z (normalized): {gripper_z}")
    
    if gripper_y_hint is None:
        gripper_y_hint = np.array([0.0, 1.0, 0.0])
    print(f"gripper_y_hint: {gripper_y_hint}")
    
    # Gram-Schmidt
    gripper_y = gripper_y_hint - np.dot(gripper_y_hint, gripper_z) * gripper_z
    print(f"gripper_y (before norm): {gripper_y}")
    print(f"gripper_y norm: {np.linalg.norm(gripper_y)}")
    
    if np.linalg.norm(gripper_y) < 1e-6:
        print("WARNING: gripper_y_hint was parallel to Z!")
        gripper_y = np.array([1.0, 0.0, 0.0])
        gripper_y = gripper_y - np.dot(gripper_y, gripper_z) * gripper_z
    
    gripper_y = gripper_y / np.linalg.norm(gripper_y)
    print(f"gripper_y (normalized): {gripper_y}")
    
    gripper_x = np.cross(gripper_y, gripper_z)
    gripper_x = gripper_x / np.linalg.norm(gripper_x)
    print(f"gripper_x: {gripper_x}")
    
    R = np.column_stack([gripper_x, gripper_y, gripper_z])
    print(f"Rotation matrix R:\n{R}")
    
    # Verify it's a valid rotation matrix
    print(f"R @ R.T (should be identity):\n{R @ R.T}")
    print(f"det(R) (should be 1): {np.linalg.det(R)}")
    
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    print(f"Output quaternion: {quat}")
    
    # Verify: convert back and check Z axis
    R_check = np.zeros(9)
    mujoco.mju_quat2Mat(R_check, quat)
    R_check = R_check.reshape(3, 3)
    print(f"Z-axis from quat (should match approach_dir): {R_check[:, 2]}")
    
    return quat


# Test with the values from the run
approach_dir = np.array([-0., 0.70710667, -0.7071069])
plate_y = np.array([1.0, 0.0, 0.0])  # Guessing plate Y for now

print("Test 1: With default gripper_y_hint (world Y)")
q1 = approach_dir_to_quat(approach_dir, gripper_y_hint=None)

print("\n" + "="*50)
print("Test 2: With plate Y as hint")
q2 = approach_dir_to_quat(approach_dir, gripper_y_hint=plate_y)




# def main():
#     env = gym.make("XArmDishwasher-v0", render_mode="human")
#     env.reset()

#     model = env.unwrapped.model
#     data = env.unwrapped.data
    
#     mujoco.mj_forward(model, data)

#     plate_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
#     plate_pos = data.site_xpos[plate_sid].copy()
#     # Orientation as rotation matrix
#     plate_mat = data.site_xmat[plate_sid].reshape(3, 3)

#     # The columns of this matrix are the site's X, Y, Z axes in world frame
#     plate_x = plate_mat[:, 0]  # Site's X-axis
#     plate_y = plate_mat[:, 1]  # Site's Y-axis  
#     plate_z = plate_mat[:, 2]  # Site's Z-axis (normal to plate surface)

#     print(f"\nPlate frame (in world coordinates):")
#     print(f"  X-axis: {plate_x}")
#     print(f"  Y-axis: {plate_y}")
#     print(f"  Z-axis (normal): {plate_z}")

#     approach_dir = -plate_z



# if __name__ == "__main__":
#     main()