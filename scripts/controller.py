import numpy as np
import mujoco
from typing import Optional, List

class DiffIKController:
    """
    A Damped Least Squares (DLS) IK controller.
    Calculates joint velocities to move the End Effector (TCP) 
    towards a target position/orientation.
    """
    def __init__(
        self, 
        model: mujoco.MjModel, 
        data: mujoco.MjData, 
        site_name: str, 
        dof_indices: Optional[List[int]] = None,
        damping: float = 1e-4, 
        step_size: float = 1.0
    ) -> None:
        self.model = model
        self.data = data
        self.site_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.dof_indices = dof_indices
        
        # Hyperparameters
        self.damping = damping  # Damping factor for DLS
        self.step_size = step_size # Scaling factor for the calculated velocity

        # Cache for Jacobian arrays (3 translation + 3 rotation = 6 DOF)
        # nv = number of degrees of freedom in the robot model
        self.jac: np.ndarray = np.zeros((6, model.nv), dtype=np.float64) 
        self.diag: np.ndarray = damping * np.eye(6, dtype=np.float64)

    def get_action(
        self, 
        target_pos: np.ndarray, 
        target_quat: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Returns delta_q (change in joint positions) to move towards target.
        """
        # calculate Error (Where are we vs where do we want to be?)
        err: np.ndarray = np.zeros(6, dtype=np.float64)
        
        # position error (dx)
        tcp_pos = self.data.site_xpos[self.site_id] # this is the TCP pos 
        # site we are controlling ^ 
        err[:3] = target_pos - tcp_pos

        # Orientation error (dr) 
        if target_quat is not None:
            current_mat = self.data.site_xmat[self.site_id]
            current_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(current_quat, current_mat)
            neg_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_negQuat(neg_quat, current_quat)
            err_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mulQuat(err_quat, target_quat, neg_quat)
            err[3:] = err_quat[1:] * 2.0
        
        # compute Jacobian (Full 6 x 19 matrix)
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)

        # slice Jacobian (Keep only the arm columns)
        # If dof_indices is provided, we extract a (6 x 7) matrix
        if self.dof_indices is not None:
            jac_active = self.jac[:, self.dof_indices]
        else:
            jac_active = self.jac

        # solve Damped Least Squares
        # dq = J_T * (J * J_T + lambda * I)^-1 * error
        jac_t = jac_active.T
        jj_t = jac_active @ jac_t + self.diag
        
        # linear system to find desired cartesian velocity
        delta_x = np.linalg.solve(jj_t, err)
        
        # Map cartesian velocity to joint velocity
        delta_q = jac_t @ delta_x

        # Return scaled step
        return delta_q * self.step_size