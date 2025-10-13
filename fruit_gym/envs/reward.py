from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import mujoco

@dataclass
class RewardConfig:
    reward_type: str = "dense"
    disappear_delay_steps: int = 16
    use_potential_rewards: bool = False
    shaping_gamma: float = 0.99
    reward_scales: dict | None = None

    def scales(self) -> dict:
        d = {
            "r_grasp": 8.0,
            "r_red": 4.0,
            "r_alignment": 1.0,
            "r_in_box": 1.0,
            "r_green_in_box_penalty": 1.0,
            "r_col": 1.0,
            "r_dist": 1.0,
            "r_attempt_close": 2.0,
            "r_bad_grasp": 2.0,
            "r_energy": 1.0,
            "r_smooth": 1.0,
            "r_gripper": 0.0,
            "r_alive": 0.0,
        }
        if self.reward_scales:
            d.update(self.reward_scales)
        return d


# -----------------------
# Utility: fruit geometry
# -----------------------

def _stem_pos(env, fruit_id: int) -> np.ndarray:
    """World pos of the stem geom for a fruit id (from env.fruit_instances)."""
    inst = env.fruit_instances.get(fruit_id, None)
    if not inst or inst.stem_geom is None:
        return None
    return env.data.geom_xpos[int(inst.stem_geom)].copy()

def _fruit_rep_pos(env, fruit_id: int) -> np.ndarray:
    """Representative world pos of the fruit (use the first fruit geom)."""
    inst = env.fruit_instances.get(fruit_id, None)
    if not inst or not inst.fruit_geoms:
        return None
    gid = int(inst.fruit_geoms[0])
    return env.data.geom_xpos[gid].copy()

def _is_point_in_gripper_box(env, point_world: np.ndarray) -> bool:
    # match your original dimensions
    BOX_HEIGHT = 0.041
    BOX_DEPTH = 0.038

    pos_left = env.data.site("left_pinch").xpos
    pos_right = env.data.site("right_pinch").xpos
    box_origin = (pos_left + pos_right) / 2
    box_R = env.data.site("long_pinch").xmat.reshape(3, 3)
    box_width = np.linalg.norm(pos_left - pos_right)

    local = box_R.T @ (point_world - box_origin)
    in_h = (-BOX_HEIGHT/2) <= local[0] <= (BOX_HEIGHT/2)
    in_w = (-box_width/2) <= local[1] <= (box_width/2)
    in_d = (-BOX_DEPTH/2) <= local[2] <= (box_width/2)
    return bool(in_h and in_w and in_d)

def _is_allowed_nonstem_contact(name: str) -> bool:
    """
    Returns True if a non-stem geom is allowed to touch finger surfaces.
    You said you've renamed those geoms to always contain 'last'.
    """
    return isinstance(name, str) and ("last" in name)

def _fruit_maps(env):
    """Build fast maps from geom ids to fruit ids and 'rep' geom ids to cache positions."""
    # Which fruits exist
    red_ids = list(getattr(env, "red_blocks", []))
    green_ids = list(getattr(env, "green_blocks", []))

    # Map stem geom id -> fruit id
    stem_to_fruit = {}
    rep_geom = {}  # fruit id -> representative fruit geom id (first fruit geom)
    for fid, inst in getattr(env, "fruit_instances", {}).items():
        if inst.stem_geom is not None:
            stem_to_fruit[int(inst.stem_geom)] = int(fid)
        if inst.fruit_geoms:
            rep_geom[int(fid)] = int(inst.fruit_geoms[0])
    return red_ids, green_ids, stem_to_fruit, rep_geom


def _rep_pos(env, fruit_id: int, rep_geom: dict):
    gid = rep_geom.get(int(fruit_id))
    if gid is None:
        return None
    return env.data.geom_xpos[gid].copy()


# -----------------------
# Privileged info (no sensors)
# -----------------------

def get_privileged_info(env) -> dict:
    """Public accessor for asymmetric RL critics."""
    return _compute_privileged_info(env)

def _compute_privileged_info(env) -> dict:
    _MAX_DIST = 1.0

    # Sites (use ids for robustness)
    pinch_sid      = env.model.site("pinch").id
    long_pinch_sid = env.model.site("long_pinch").id
    left_sid       = env.model.site("left_pinch").id
    right_sid      = env.model.site("right_pinch").id

    tcp_pos  = env.data.site_xpos[long_pinch_sid]
    pinch_R  = env.data.site_xmat[pinch_sid].reshape(3, 3)
    gripper_y = pinch_R[:, 1]

    red_ids, green_ids, stem_to_fruit, rep_geom = _fruit_maps(env)

    info = {
        "min_red_dist": _MAX_DIST,
        "radial_dist": 0.0,
        "good_grasp": False,
        "bad_grasp": False,
        "collision_detected": False,
        "red_stems_in_box_count": 0,
        "green_stems_in_box_count": 0,
        "left_finger_contacts": 0,
        "right_finger_contacts": 0,
        "total_displacement": 0.0,
        "grasped_idx": None,
        "grasped_unripe_idx": None,
    }

    # Nearest red stem (compute from stem geom positions)
    if red_ids:
        dists = {}
        for rid in red_ids:
            inst = env.fruit_instances.get(rid)
            if not inst or inst.stem_geom is None:
                continue
            s = env.data.geom_xpos[int(inst.stem_geom)]
            dists[rid] = np.linalg.norm(s - tcp_pos)
        if dists:
            closest = min(dists, key=dists.get)
            info["min_red_dist"] = float(dists[closest])
            s = env.data.geom_xpos[int(env.fruit_instances[closest].stem_geom)]
            vec = s - tcp_pos
            info["radial_dist"] = abs(np.dot(vec, gripper_y))
    else:
        info["min_red_dist"] = 0.0
        info["radial_dist"] = 0.0

    # Red stems inside gripper box
    def _in_box(p):
        BOX_H, BOX_D = 0.041, 0.038
        pos_left  = env.data.site_xpos[left_sid]
        pos_right = env.data.site_xpos[right_sid]
        origin = (pos_left + pos_right) / 2
        R = env.data.site_xmat[long_pinch_sid].reshape(3, 3)
        w = np.linalg.norm(pos_left - pos_right)
        local = R.T @ (p - origin)
        return (-BOX_H/2 <= local[0] <= BOX_H/2) and (-w/2 <= local[1] <= w/2) and (-BOX_D/2 <= local[2] <= BOX_D/2)

    for rid in red_ids:
        inst = env.fruit_instances.get(rid)
        if not inst or inst.stem_geom is None:
            continue
        if _in_box(env.data.geom_xpos[int(inst.stem_geom)]):
            info["red_stems_in_box_count"] += 1

    # Any green stem in the box?
    green_in_box = False
    for gid in green_ids:
        inst = env.fruit_instances.get(gid)
        if not inst or inst.stem_geom is None:
            continue
        if _in_box(env.data.geom_xpos[int(inst.stem_geom)]):
            green_in_box = True
            break
    info["green_stems_in_box_count"] = 1 if green_in_box else 0

    # Contacts (ID-based): detect good/bad grasp + collisions
    left_contacts = right_contacts = 0
    left_red_stem_contacts = set()
    right_red_stem_contacts = set()
    left_green_stem_contacts = set()
    right_green_stem_contacts = set()
    collision = False

    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1 = c.geom1
        g2 = c.geom2
        name1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
        name2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""

        # Collision penalty: finger vs non-stem, non-allowed fruit part
        is_finger_contact = "finger" in name1 or "finger" in name2
        is_fruit_contact = "fruit" in name1 or "fruit" in name2
        if is_finger_contact and is_fruit_contact:
            # Check if contact is with an allowed geom or a stem (handled separately)
            g1_ok = _is_allowed_nonstem_contact(name1) or g1 in stem_to_fruit
            g2_ok = _is_allowed_nonstem_contact(name2) or g2 in stem_to_fruit
            if not (g1_ok and g2_ok):
                collision = True

        # Grasp detection: finger vs stem
        # LEFT finger
        if "left_finger_inner" in (name1, name2):
            left_contacts += 1
            other_gid = g1 if name2 == "left_finger_inner" else g2
            if other_gid in stem_to_fruit:
                sidx = stem_to_fruit[other_gid]
                if sidx in red_ids:
                    left_red_stem_contacts.add(sidx)
                elif sidx in green_ids:
                    left_green_stem_contacts.add(sidx)

        # RIGHT finger
        if "right_finger_inner" in (name1, name2):
            right_contacts += 1
            other_gid = g1 if name2 == "right_finger_inner" else g2
            if other_gid in stem_to_fruit:
                sidx = stem_to_fruit[other_gid]
                if sidx in red_ids:
                    right_red_stem_contacts.add(sidx)
                elif sidx in green_ids:
                    right_green_stem_contacts.add(sidx)

    # Analyze contacts to determine grasp type
    common_red_stems = left_red_stem_contacts.intersection(right_red_stem_contacts)
    common_green_stems = left_green_stem_contacts.intersection(right_green_stem_contacts)

    good_grasp = len(common_red_stems) == 1
    bad_grasp = len(common_green_stems) > 0

    info["left_finger_contacts"] = left_contacts
    info["right_finger_contacts"] = right_contacts
    info["collision_detected"] = collision
    info["good_grasp"] = good_grasp
    info["bad_grasp"] = bad_grasp
    if good_grasp:
        info["grasped_idx"] = common_red_stems.pop()
    if bad_grasp:
        info["grasped_unripe_idx"] = common_green_stems.pop()


    # Displacement (using representative fruit geom)
    total = 0.0
    for gid in green_ids:
        cur = _rep_pos(env, gid, rep_geom)
        init = getattr(env, "green_positions", {}).get(gid, cur)
        if cur is not None and init is not None:
            total += np.linalg.norm(cur - init)
    for rid in red_ids:
        cur = _rep_pos(env, rid, rep_geom)
        init = getattr(env, "red_positions", {}).get(rid, cur)
        if cur is not None and init is not None:
            total += np.linalg.norm(cur - init)
    info["total_displacement"] = float(total)
    return info
# -----------------------


def _phi_red(priv: dict) -> float:
    d = priv["min_red_dist"]
    if not np.isfinite(d):
        return 0.0
    return float(-d)

def _phi_align(priv: dict) -> float:
    if not np.isfinite(priv["min_red_dist"]):
        return 0.0
    return float(-priv["radial_dist"])


# -----------------------
# Main reward
# -----------------------

def compute_reward(env, action: np.ndarray) -> tuple[float, dict]:
    cfg: RewardConfig = getattr(env, "reward_cfg", RewardConfig())
    scales = cfg.scales()

    priv = _compute_privileged_info(env)

    if cfg.use_potential_rewards:
        phi_red_now = _phi_red(priv)
        phi_align_now = _phi_align(priv)
        prev_phi_red = getattr(env, "_prev_phi_red", 0.0)
        prev_phi_align = getattr(env, "_prev_phi_align", 0.0)
        r_red = cfg.shaping_gamma * phi_red_now - prev_phi_red
        r_alignment = cfg.shaping_gamma * phi_align_now - prev_phi_align
        env._prev_phi_red = phi_red_now
        env._prev_phi_align = phi_align_now
    else:
        d = priv["min_red_dist"]
        r_red = -np.tanh(20.0 * d) if np.isfinite(d) else 0.0
        r_alignment = -np.tanh(60.0 * priv["radial_dist"]) if np.isfinite(d) else 0.0

    r_in_box = 0.0 if (priv["red_stems_in_box_count"] == 1 and priv["green_stems_in_box_count"] == 0) else -1.0
    r_green_in_box_penalty = -1.0 if priv["green_stems_in_box_count"] > 0 else 0.0
    r_col = -1.0 if priv["collision_detected"] else 0.0
    r_dist = -np.tanh(5.0 * priv["total_displacement"])

    r_energy = -np.linalg.norm(action[:-1])
    r_smooth = -np.linalg.norm(action[:-1] - getattr(env, "prev_action", np.zeros_like(action))[:-1])

    r_gripper = -1.0 if (
        np.array_equal(getattr(env, "gripper_vec"), env.gripper_dict["closing"]) or
        np.array_equal(getattr(env, "gripper_vec"), env.gripper_dict["opening"])
    ) else 0.0

    r_attempt_close = 0.0
    if np.array_equal(getattr(env, "gripper_vec"), env.gripper_dict["closing"]):
        if priv["min_red_dist"] < 0.03:
            r_attempt_close = 1.0

    # Grasp logic
    r_grasp = 0.0
    r_bad_grasp = -float(priv["bad_grasp"])

    # Successful grasp of a ripe fruit
    gid = priv.get("grasped_idx", None)
    if priv["good_grasp"] and (not priv["bad_grasp"]) and gid is not None:
        cur = _fruit_rep_pos(env, gid)
        init = env.red_positions.get(gid, cur)
        if cur is not None and init is not None and np.linalg.norm(cur - init) < 0.05:
            r_grasp = 1.0
            if gid not in env._grasped_pending:
                env._ripe_fruits_picked = getattr(env, "_ripe_fruits_picked", 0) + 1
                env._grasped_pending.add(gid)
                env._pending_removals[gid] = int(cfg.disappear_delay_steps)

    # Log grasp attempt on an unripe fruit
    unripe_gid = priv.get("grasped_unripe_idx", None)
    if priv["bad_grasp"] and unripe_gid is not None:
        if not hasattr(env, "_grasped_unripe_pending"):
            env._grasped_unripe_pending = set()
        if unripe_gid not in env._grasped_unripe_pending:
            env._unripe_fruits_picked = getattr(env, "_unripe_fruits_picked", 0) + 1
            env._grasped_unripe_pending.add(unripe_gid)

    r_alive = -1.0

    rewards = {
        "r_grasp": r_grasp,
        "r_red": r_red,
        "r_alignment": r_alignment,
        "r_in_box": r_in_box,
        "r_green_in_box_penalty": r_green_in_box_penalty,
        "r_col": r_col,
        "r_dist": r_dist,
        "r_attempt_close": r_attempt_close,
        "r_bad_grasp": r_bad_grasp,
        "r_energy": r_energy,
        "r_smooth": r_smooth,
        "r_gripper": r_gripper,
        "r_alive": r_alive,
    }
    rewards = {k: v * scales[k] for k, v in rewards.items()}
    reward = float(np.clip(sum(rewards.values()), -1e4, 1e4))
    success = (len(getattr(env, "red_blocks", [])) == 0)

    info = dict(rewards)
    info["ripe_fruits_picked"] = getattr(env, "_ripe_fruits_picked", 0)
    info["unripe_fruits_picked"] = getattr(env, "_unripe_fruits_picked", 0)
    info["success"] = success
    return reward, info