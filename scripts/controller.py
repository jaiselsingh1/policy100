from typing import Optional, Tuple, List 
from dataclasses import dataclass 
import mujoco
import numpy as np 

@dataclass
class IKConfig:
    damping: float = 1e-3          # Damping factor for singularity robustness
    max_delta_q: float = 0.1       # Maximum joint position change per step (rad)
    pos_gain: float = 1.0          # Position error gain
    ori_gain: float = 0.5          # Orientation error gain
    nullspace_gain: float = 0.1    # Nullspace projection gain
    tolerance_pos: float = 0.005   # Position convergence threshold (m)
    tolerance_ori: float = 0.05    # Orientation convergence threshold (rad)

class DiffIKController:
    """Differential Inverse Kinematics Controller using Damped Least Squares."""
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_name: str = "link_tcp",
        joint_names: Optional[List[str]] = None,
        config: Optional[IKConfig] = None,
    ):
        
        self.model = model 
        self.data = data 
        self.config = config or IKConfig()

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(f"Site {site_name} not found")
        
        if joint_names is None:
            # Detect 7 joints typically used in xarm
            joint_names = [f"joint{i}" for i in range(1, 8)]

        self.joint_names = joint_names
        self.n_joints = len(joint_names)

        # Cache joint IDs and addresses
        self.joint_ids = []
        self.qpos_indices = []
        self.dof_indices = []
        self.joint_limits = []

        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found in model")
            
            self.joint_ids.append(jid)
            self.qpos_indices.append(model.jnt_qposadr[jid])
            self.dof_indices.append(model.jnt_dofadr[jid])
            self.joint_limits.append(model.jnt_range[jid].copy())
        
        self.qpos_indices = np.array(self.qpos_indices, dtype=np.int64)
        self.dof_indices = np.array(self.dof_indices, dtype=np.int64)
        self.joint_limits = np.array(self.joint_limits)
        
        # Preallocate Jacobian matrices
        self.jac_pos = np.zeros((3, model.nv))
        self.jac_rot = np.zeros((3, model.nv))
        
        # Default nullspace configuration (center of joint ranges)
        self.nullspace_target = 0.5 * (self.joint_limits[:, 0] + self.joint_limits[:, 1])

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns pos (3,) and quat (4,) (w,x,y,z)"""
        pos = self.data.site_xpos[self.site_id].copy()
        rot_mat = self.data.site_xmat[self.site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot_mat.flatten())
        return pos, quat
    
    def get_current_joints(self) -> np.ndarray:
        return self.data.qpos[self.qpos_indices].copy()
    
    def _compute_jacobian(self) -> Tuple[np.ndarray, np.ndarray]:
        mujoco.mj_jacSite(
            self.model, self.data,
            self.jac_pos, self.jac_rot,
            self.site_id
        )
        return self.jac_pos[:, self.dof_indices], self.jac_rot[:, self.dof_indices]
    
    def _quaternion_error(self, q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        # difference quarternion calculator
        # Conceptually: q_diff = q_target * q_current_inverse
        q_current_inv = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        
        q_error = np.zeros(4)
        mujoco.mju_mulQuat(q_error, q_target, q_current_inv)
        
        # handle the double cover problem
        # Ensure w (scalar part) is positive to take the shortest path
        if q_error[0] < 0:
            q_error = -q_error
            
        # convert to axis angle (Ax = b doesn't understand quaternions)
        angle = 2.0 * np.arccos(np.clip(q_error[0], -1.0, 1.0))
        
        # signularity check for small angles 
        if angle < 1e-6:
            return np.zeros(3)
        
        axis = q_error[1:] / np.sin(angle / 2.0)
        
        # return scaled axis 
        return angle * axis

    
    def compute(
            self, 
            target_pos: np.ndarray, 
            target_quat: Optional[np.ndarray] = None, 
            use_nullspace: bool = True) -> np.ndarray:
        """computes the joint velocity needed to take end effector to the target"""

        current_pos, current_quat = self.get_ee_pose()
        current_q = self.get_current_joints()
        jac_pos, jac_rot = self._compute_jacobian()

        pos_error = target_pos - current_pos 
        
        if target_quat is not None:
            ori_error = self._quaternion_error(target_quat, current_quat)
            J = np.vstack([
                self.config.pos_gain * jac_pos,
                self.config.ori_gain * jac_rot
            ])

            error = np.concatenate([
                self.config.pos_gain * pos_error,
                self.config.ori_gain * ori_error
            ])
        else:
            # position only
            J = self.config.pos_gain * jac_pos
            error = self.config.pos_gain * pos_error

        lambda_sq = self.config.damping ** 2
        JJT = J @ J.T

        # add damping to the diagnol for stability reasons 
        I_damp = np.eye(JJT.shape[0]) * lambda_sq

        # Solve linear system for the "Cartesian" force/velocity
        # (J @ J.T + lam*I) * x = error 
        velocity_cartesian = np.linalg.solve(JJT + I_damp, error)
        
        # Map back to joint space
        dq = J.T @ velocity_cartesian

        # nullspace projection 
        # this pulls the arm towards the "mean" joint configuration to avoid awkward angles
        if use_nullspace:
            # Calculate pseudoinverse J_pinv = J.T @ inv(JJT + I_damp)
            J_pinv = J.T @ np.linalg.inv(JJT + I_damp)
            
            # Projector N = I - J_pinv @ J
            N = np.eye(self.n_joints) - J_pinv @ J
            
            # Secondary error: difference between current q and comfortable "neutral" q
            null_error = self.config.nullspace_gain * (self.nullspace_target - current_q)
            
            # Project this error into the nullspace so it doesn't fight the main task
            dq += N @ null_error

        # clip the maximum that the arm can move for safety checks
        norm = np.linalg.norm(dq)
        if norm > self.config.max_delta_q:
            dq *= (self.config.max_delta_q / norm)

        return dq
    
    def step_toward(
        self, 
        target_pos: np.ndarray, 
        target_quat: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, float]:
        """
        Computes the next joint configuration (clipped to limits) and returns current errors.
        Returns: (new_q, pos_error, ori_error)
        """
        # Get current state
        current_q = self.get_current_joints()
        
        # Calculate the delta
        dq = self.compute(target_pos, target_quat)
        
        # Integrate
        new_q = current_q + dq
        
        # Enforce Joint Limits
        new_q = np.clip(new_q, self.joint_limits[:, 0], self.joint_limits[:, 1])

        # Calculate errors for reporting/convergence checking
        # (We recalculate these based on current state to be accurate)
        curr_pos, curr_quat = self.get_ee_pose()
        pos_error = np.linalg.norm(target_pos - curr_pos)
        
        ori_error = 0.0
        if target_quat is not None:
            # Reuse internal error calc for consistency
            ori_vec = self._quaternion_error(target_quat, curr_quat)
            ori_error = np.linalg.norm(ori_vec)

        return new_q, pos_error, ori_error
    

    def is_converged(
        self, 
        target_pos: np.ndarray, 
        target_quat: Optional[np.ndarray] = None
    ) -> bool:
        """
        Checks if the end-effector is within the tolerances defined in IKConfig.
        """
        curr_pos, curr_quat = self.get_ee_pose()
        
        # Check Position
        pos_dist = np.linalg.norm(target_pos - curr_pos)
        if pos_dist > self.config.tolerance_pos:
            return False
        
        # Check Orientation
        if target_quat is not None:
            ori_vec = self._quaternion_error(target_quat, curr_quat)
            ori_dist = np.linalg.norm(ori_vec)
            if ori_dist > self.config.tolerance_ori:
                return False
                
        return True
    
    def get_joint_solution(
        self, 
        target_pos: np.ndarray, 
        target_quat: Optional[np.ndarray] = None,
        max_iters: int = 200,
        seed_q: Optional[np.ndarray] = None) -> Optional[np.ndarray]:

        original_q = self.get_current_joints().copy()
        q = seed_q.copy() if seed_q is not None else original_q.copy()
        success = False

        for _ in range(max_iters):
            self.data.qpos[self.qpos_indices] = q
            mujoco.mj_forward(self.model, self.data)

            if self.is_converged(target_pos, target_quat):
                success = True
                break

            dq = self.compute(target_pos, target_quat)
            q += dq
            q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

        self.data.qpos[self.qpos_indices] = original_q
        mujoco.mj_forward(self.model, self.data)

        return q if success else None

def make_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Creates a quaternion (w, x, y, z) from Euler angles (xyz convention).
    Useful for defining target orientations.
    """
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "xyz")
    return q
