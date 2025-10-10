# tune_scales_from_real.py
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from fruit_gym import envs
import os
import cv2
os.environ["MUJOCO_GL"] = "egl"

REAL_TRAJ = "real_box_path.npz"      # change if needed
SIM_ENV_ID = "PickMultiStrawbHardEnv"

# --- loss weighting & options ---
W_ROT = 1.0                # rotation error weight (0.2–2.0 is typical)
ACTION_DELAY_STEPS = 0     # try 1–2 if sim is too "snappy"
MAX_STEPS_BUFFER = 10      # extra steps to avoid TimeLimit before the log ends

# --- search grids around 0.06 ---
POS_GRID  = [0.0035, 0.004, 0.0045, 0.0050, 0.0055, 0.0060, 0.0065]
ROT_GRID  = [0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075]

def quat_to_R(q):
    x,y,z,w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=float)

def rot_angle_between(q_real, q_sim):
    Rr = quat_to_R(q_real); Rs = quat_to_R(q_sim)
    Rerr = Rr @ Rs.T
    tr = float(np.trace(Rerr))
    tr = max(-1.0, min(3.0, tr))
    return np.arccos((tr - 1.0)/2.0)

def make_env_with_scales(pos_scale, rot_scale):
    env = gym.make(
        SIM_ENV_ID,
        cameras=['wrist2', 'front'],
        width=480,
        height=480,
        physics_dt=0.001,
        ee_dof=6,
        pos_scale=float(pos_scale),
        rot_scale=float(rot_scale),
    )
    return env

def run_sim_and_loss(traj_path, pos_scale, rot_scale, delay_steps=0, w_rot=1.0):
    data = np.load(traj_path, allow_pickle=True)
    A = data["action"]            # [T, 7] actions logged on the real run
    P_real = data["tcp_pose"]     # [T, 7] measured EE pose from the real run

    env = make_env_with_scales(pos_scale, rot_scale)
    env = TimeLimit(env, max_episode_steps=len(A) + MAX_STEPS_BUFFER)
    obs, info = env.reset()
    print(f"successfully reset sim env with pos_scale={pos_scale} rot_scale={rot_scale}")

    P_sim = []
    # optional action delay buffer
    if delay_steps > 0:
        buf = [np.zeros_like(A[0]) for _ in range(delay_steps)]
    else:
        buf = []

    for a in A:
        a_to_apply = a
        if buf:
            buf.append(a)
            a_to_apply = buf.pop(0)
        obs, _, term, trunc, _ = env.step(a_to_apply)
        cv2.imshow("sim", obs['images']['front'])
        cv2.waitKey(1)
        P_sim.append(obs['state']['tcp_pose'].copy())
        if term or trunc:
            break
    env.close()
    P_sim = np.asarray(P_sim)
    T = min(len(P_sim), len(P_real))
    if T == 0:
        return float("inf")

    P_sim  = P_sim[:T]
    P_real = P_real[:T]

    # position loss
    pos_err = P_sim[:, :3] - P_real[:, :3]
    pos_loss = float(np.sum(pos_err**2))

    # rotation loss (sum of squared rotation angles)
    rot_loss = 0.0
    for qr, qs in zip(P_real[:, 3:7], P_sim[:, 3:7]):
        rot_loss += rot_angle_between(qr, qs)**2

    return pos_loss + w_rot * rot_loss

def grid_search(traj_path):
    best = (float("inf"), None)
    tried = 0
    for ps in POS_GRID:
        for rs in ROT_GRID:
            tried += 1
            J = run_sim_and_loss(traj_path, ps, rs,
                                 delay_steps=ACTION_DELAY_STEPS,
                                 w_rot=W_ROT)
            print(f"[{tried:03d}] loss={J:.6f}  pos_scale={ps:.5f}  rot_scale={rs:.5f}")
            if J < best[0]:
                best = (J, (ps, rs))
                print("   ✓ new best")
    return best

def main():
    best_loss, (best_ps, best_rs) = grid_search(REAL_TRAJ)
    print("\n=== Best scales ===")
    print(f"pos_scale={best_ps:.5f}, rot_scale={best_rs:.5f}   loss={best_loss:.6f}")

    # Optional: quick confirm run
    confirm = run_sim_and_loss(REAL_TRAJ, best_ps, best_rs,
                               delay_steps=ACTION_DELAY_STEPS, w_rot=W_ROT)
    print(f"Confirm loss: {confirm:.6f}")

if __name__ == "__main__":
    main()
