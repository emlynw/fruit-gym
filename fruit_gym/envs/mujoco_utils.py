import mujoco

def mj_name(model, objtype, objid) -> str:
    try:
        return mujoco.mj_id2name(model, objtype, int(objid)) or ""
    except mujoco.Error:
        return ""

def geom_name(model, gid: int) -> str:
    return mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)

def mat_name_for_geom(model, gid: int) -> str:
    mid = int(model.geom_matid[gid])
    if mid < 0:
        return ""
    return mj_name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mid)

def body_parent(model, bid: int) -> int:
    return int(model.body_parentid[bid])

###### Utils for deleting picked strawbs ######

def set_inactive_properties_recursive(model, body_id: int):
    start = model.body_geomadr[body_id]
    count = model.body_geomnum[body_id]
    for k in range(count):
        gid = start + k
        model.geom_group[gid] = 3
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
    # recurse
    for child_id in range(model.nbody):
        if model.body_parentid[child_id] == body_id:
            set_inactive_properties_recursive(model, child_id)

def hide_fruit_by_instance(env, fruit_id: int):
    """
    Hide all geoms that belong to the fruit instance (uses env.fruit_instances[fruit_id]).
    """
    inst = env.fruit_instances.get(fruit_id, None)
    if not inst:
        return
    # deactivate all fruit geoms
    for gid in getattr(inst, "fruit_geoms", []):
        gid = int(gid)
        env.model.geom_group[gid] = 3
        env.model.geom_contype[gid] = 0
        env.model.geom_conaffinity[gid] = 0
    # if you also want to nuke the whole body tree, you can find a body via the geom owner:
    # (optional safety — only if your assets are one-fruit-per-body subtrees)
    try:
        from mujoco import mjtObj, mj_id2name
        if inst.fruit_geoms:
            any_gid = int(inst.fruit_geoms[0])
            bid = int(env.model.geom_bodyid[any_gid])
            set_inactive_properties_recursive(env.model, bid)
    except Exception:
        pass

def apply_removal(env, idx: int):
    hide_fruit_by_instance(env, idx)
    if hasattr(env, "active_indices"):
        try:
            import numpy as _np
            env.active_indices = _np.delete(env.active_indices, _np.where(env.active_indices == idx))
        except Exception:
            pass
    if hasattr(env, "red_blocks") and (idx in env.red_blocks):
        env.red_blocks.remove(idx)

def tick_removal_timers(env):
    if not getattr(env, "_pending_removals", None):
        return
    for k in list(env._pending_removals.keys()):
        env._pending_removals[k] -= 1
    due = [k for k, t in env._pending_removals.items() if t <= 0]
    for idx in due:
        apply_removal(env, idx)
        env._pending_removals.pop(idx, None)
        env._grasped_pending.discard(idx)
