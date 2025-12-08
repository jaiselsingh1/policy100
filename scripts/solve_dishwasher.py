import time
import numpy as np
import gymnasium as gym
import mujoco
import policy100.envs

from controller import DiffIKController, IKConfig



import time
import numpy as np
import gymnasium as gym
import mujoco

from controller import DiffIKController, IKConfig


def step_ik(env, ik, target_pos, steps):
    model = env.unwrapped.model
    data = env.unwrapped.data

    for _ in range(steps):
        dq = ik.compute(target_pos, target_quat=None, use_nullspace=False)
        q = ik.get_current_joints()
        new_q = q + dq
        new_q = np.clip(new_q, ik.joint_limits[:, 0], ik.joint_limits[:, 1])

        data.qpos[ik.qpos_indices] = new_q
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        env.render()


def set_gripper_qpos(env, value, steps):
    model = env.unwrapped.model
    data = env.unwrapped.data

    if hasattr(env.unwrapped, "_gripper_qpos_idx"):
        idx = env.unwrapped._gripper_qpos_idx
    else:
        raise RuntimeError("env.unwrapped._gripper_qpos_idx not defined; set it in the env")

    for _ in range(steps):
        data.qpos[idx] = value
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        env.render()


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

    hover_pos = plate_pos + np.array([0.0, 0.0, 0.15])
    grasp_pos = plate_pos + np.array([0.0, 0.0, 0.025])
    lift_pos = hover_pos

    for _ in range(10):
        env.render()

    # approach hover above plate
    step_ik(env, ik, hover_pos, steps=300)

    # move down to grasp pose
    step_ik(env, ik, grasp_pos, steps=200)

    # close gripper
    set_gripper_qpos(env, value=1.0, steps=150)

    # lift plate
    step_ik(env, ik, lift_pos, steps=300)

    time.sleep(1.0)
    env.close()


if __name__ == "__main__":
    main()


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
#             ori_gain=0.0,
#             nullspace_gain=0.0,
#             tolerance_pos=0.003,
#             tolerance_ori=0.1,
#         ),
#     )

#     plate_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
#     if plate_sid == -1:
#         raise RuntimeError("site 'plate_center' not found")

#     mujoco.mj_forward(model, data)
#     plate_pos = data.site_xpos[plate_sid].copy()
#     target_pos = plate_pos + np.array([0.0, 0.0, 0.10])

#     for _ in range(10):
#         env.render()

#     for t in range(400):
#         dq = ik.compute(target_pos, target_quat=None, use_nullspace=False)
#         q = ik.get_current_joints()
#         new_q = q + dq
#         new_q = np.clip(new_q, ik.joint_limits[:, 0], ik.joint_limits[:, 1])

#         data.qpos[ik.qpos_indices] = new_q
#         mujoco.mj_forward(model, data)
#         env.render()

#     time.sleep(1.0)
#     env.close()


# if __name__ == "__main__":
#     main()

