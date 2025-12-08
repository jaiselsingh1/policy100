import time
import numpy as np
import gymnasium as gym
import mujoco
import policy100.envs

from controller import DiffIKController, IKConfig

def main():
    env = gym.make("XArmDishwasher-v0", render_mode="human")
    obs, info = env.reset()

    model = env.unwrapped.model
    data = env.unwrapped.data

    ik = DiffIKController(
        model,
        data,
        site_name="tcp_site",
        config=IKConfig(
            damping=1e-3,
            max_delta_q=0.05,
            pos_gain=1.0,
            ori_gain=0.0,
            nullspace_gain=0.0,
            tolerance_pos=0.003,
            tolerance_ori=0.1,
        ),
    )

    plate_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
    if plate_sid == -1:
        raise RuntimeError("site 'plate_center' not found")

    mujoco.mj_forward(model, data)
    plate_pos = data.site_xpos[plate_sid].copy()
    target_pos = plate_pos + np.array([0.0, 0.0, 0.10])

    for _ in range(10):
        env.render()

    for t in range(400):
        dq = ik.compute(target_pos, target_quat=None, use_nullspace=False)
        q = ik.get_current_joints()
        new_q = q + dq
        new_q = np.clip(new_q, ik.joint_limits[:, 0], ik.joint_limits[:, 1])

        data.qpos[ik.qpos_indices] = new_q
        mujoco.mj_forward(model, data)
        env.render()

    time.sleep(1.0)
    env.close()


if __name__ == "__main__":
    main()








# GRIPPER_OPEN = -1.0

# def send_pd_action(env, target_q, gripper):
#     m = env.unwrapped.model
#     d = env.unwrapped.data

#     arm_q_idx = env.unwrapped._arm_qpos_idx
#     if hasattr(env.unwrapped, "_arm_ctrl_idx"):
#         arm_ctrl_idx = env.unwrapped._arm_ctrl_idx
#     else:
#         arm_ctrl_idx = np.arange(7, dtype=int)

#     q = d.qpos[arm_q_idx]
#     qdot = d.qvel[arm_q_idx]

#     Kp = 10.0
#     Kd = 1.0
#     u = Kp * (target_q - q) - Kd * qdot

#     action = np.zeros(env.action_space.shape[0], dtype=np.float32)
#     low = env.action_space.low[arm_ctrl_idx]
#     high = env.action_space.high[arm_ctrl_idx]
#     u = np.clip(u, low, high)

#     action[arm_ctrl_idx] = u
#     action[-1] = gripper

#     return env.step(action)


# def main():
#     env = gym.make("XArmDishwasher-v0", render_mode="human")
#     obs, info = env.reset()

#     model = env.unwrapped.model
#     data = env.unwrapped.data

#     ik = DiffIKController(
#         model,
#         data,
#         site_name="tcp_site",
#         config=IKConfig(
#             damping=1e-3,
#             max_delta_q=0.05,
#             pos_gain=1.0,
#             ori_gain=0.0,   # orientation off for step 1
#             tolerance_pos=0.003,
#             tolerance_ori=0.1,
#         ),
#     )

#     plate_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
#     if plate_sid == -1:
#         raise RuntimeError("site 'plate_center' not found")

#     plate_pos = data.site_xpos[plate_sid].copy()
#     target_pos = plate_pos + np.array([0.0, 0.0, 0.20])  # hover above plate

#     for _ in range(10):
#         env.render()

#     for t in range(400):
#         dq = ik.compute(target_pos, target_quat=None)
#         current_q = ik.get_current_joints()
#         target_q = current_q + dq

#         obs, reward, terminated, truncated, info = send_pd_action(
#             env, target_q, GRIPPER_OPEN
#         )
#         env.render()

#         if terminated or truncated:
#             break

#     time.sleep(1.0)
#     env.close()


# if __name__ == "__main__":
#     main()
