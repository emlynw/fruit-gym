from .base import Randomiser
from pathlib import Path
import mujoco

# ---------------- helpers ----------------

def _move_unique_assets(dest: mujoco.MjSpec, src: mujoco.MjSpec):
    """Copy meshes/materials/… from *src* → *dest* if the *name* is new."""
    for coll in ("meshes", "textures", "materials", "skins", "hfields"):
        dlist = getattr(dest, coll)
        slist = getattr(src, coll)
        existing = {a.name for a in dlist if a.name}
        for a in list(slist):
            if a.name and a.name not in existing:
                dlist.append(a)
                slist.remove(a)

def _iter_nodes(node):
    yield node
    for b in getattr(node, "bodies", []):
        yield from _iter_nodes(b)
    for f in getattr(node, "frames", []):
        yield from _iter_nodes(f)

def _spec_has_fruit(vine_spec: mujoco.MjSpec) -> bool:
    """True if *any* geom name in this spec starts with 'fruit' (case-insensitive)."""
    for node in _iter_nodes(vine_spec.worldbody):
        for g in getattr(node, "geoms", []):
            nm = (getattr(g, "name", "") or "").lower()
            if nm.startswith("fruit"):
                return True
    return False

def _attach_with_suffix(root_spec: mujoco.MjSpec,
                        vine_spec: mujoco.MjSpec,
                        idx: int,
                        mount_prefix: str):
    """Create a mount frame and attach the vine body with a unique suffix.

    We tag as '_leaves_{idx}' if the vine_spec has NO 'fruit*' geoms.
    """
    is_leaves = not _spec_has_fruit(vine_spec)
    frame = root_spec.worldbody.add_frame(name=f"{mount_prefix}{idx}")
    suffix = f"_leaves_{idx}" if is_leaves else f"_{idx}"
    frame.attach_body(vine_spec.worldbody.bodies[0], suffix=suffix)

# ---------------- spawner ----------------

class SpawnerRandomiser(Randomiser):
    """
    Attach N copies of randomly chosen plant XMLs under unique frames.

    Guarantees at least one `fruit*` geom exists after spawning by
    conditionally adding a fallback vine (see `ensure_min_fruit`).

    Parameters
    ----------
    strawb_xml / xml_choices
        Either a single legacy path or a list of candidate XMLs.
    min_count / max_count
        Random integer in [min_count, max_count] is spawned first.
    mount_prefix
        Name prefix for each mount frame.
    ensure_min_fruit
        Minimum number of `fruit*` geoms required. If not met, a fallback
        vine (or two) is added.
    fallback_names
        Tuple of candidate filenames to use when adding fallback fruit vines.
        These are resolved relative to the directory of the first choice.
        Default: ("strawb_fork_double.xml", "strawb_fork.xml", "strawb_stiff.xml")
        Logic: pick one uniformly; if 'strawb_stiff.xml' is picked we add two.
    """

    affects_spec = True
    needs_ctx = False

    def __init__(
        self,
        strawb_xml: Path | str | None = None,
        xml_choices: list[Path | str] | None = None,
        min_count: int = 4,
        max_count: int = 8,
        mount_prefix: str = "vine_",
        ensure_min_fruit: int = 1,
        fallback_names: tuple[str, ...] = (
            "strawb_fork_double.xml",
            "strawb_fork.xml",
            "strawb_stiff.xml",
        ),
    ):
        if xml_choices is None:
            if strawb_xml is None:
                raise ValueError("Provide either strawb_xml or xml_choices")
            self._choices = [Path(strawb_xml)]
        else:
            self._choices = [Path(p) for p in xml_choices]

        for p in self._choices:
            if not p.exists():
                raise FileNotFoundError(p)

        self._min = int(min_count)
        self._max = int(max_count)
        self._mount_prefix = mount_prefix
        self._ensure_min_fruit = int(ensure_min_fruit)
        self._fallback_names = tuple(fallback_names)

    # ------------------------------------------------------------------ #

    def apply(self, *, spec, model, data, rng, ctx=None):
        if spec is None:
            raise ValueError("SpawnerRandomiser needs a spec when affects_spec=True")

        # Start fresh: remove old mount frames
        spec.worldbody.frames.clear()

        # random initial count
        count = int(rng.integers(self._min, self._max + 1))

        fruit_seen = 0
        next_idx = 0
        base_dir = self._choices[0].parent

        print(f"[Spawner] spawning {count} vines")

        # 1) primary spawns
        for i in range(count):
            choice_path = Path(rng.choice(self._choices))
            print(f"[Spawner]  -> pick {choice_path.name}")
            vine_spec = mujoco.MjSpec.from_file(str(choice_path))

            # bring over any unique assets (usually no-op in your new setup)
            _move_unique_assets(spec, vine_spec)

            # track whether this vine contains a fruit geom
            if _spec_has_fruit(vine_spec):
                fruit_seen += 1

            # attach
            _attach_with_suffix(spec, vine_spec, next_idx, self._mount_prefix)
            next_idx += 1

        # 2) ensure minimum number of fruits present
        if fruit_seen < self._ensure_min_fruit:
            need = self._ensure_min_fruit - fruit_seen
            print(f"[Spawner] Need {need} more fruit instance(s); adding fallback vine(s).")

            # pick a fallback file name
            pick = rng.choice(self._fallback_names)
            # try to load it; if missing, try others
            fallback_list = [pick, *[n for n in self._fallback_names if n != pick]]

            added_fruit = 0
            for name in fallback_list:
                path = base_dir / name
                if not path.exists():
                    continue

                # how many to add for this choice?
                to_add = 2 if name == "strawb_stiff.xml" and need - added_fruit >= 2 else 1

                for _ in range(to_add):
                    vine_spec = mujoco.MjSpec.from_file(str(path))
                    _move_unique_assets(spec, vine_spec)
                    # attach and count if it truly has fruit geoms
                    if _spec_has_fruit(vine_spec):
                        added_fruit += 1
                    _attach_with_suffix(spec, vine_spec, next_idx, self._mount_prefix)
                    print(f"[Spawner]  -> fallback add {name}")
                    next_idx += 1

                if added_fruit >= need:
                    break

            if added_fruit < need:
                print(f"[Spawner][WARN] Could not meet fruit minimum "
                      f"({fruit_seen}+{added_fruit}<{self._ensure_min_fruit}). "
                      f"Check fallback files exist in {base_dir}.")

        # (Optional) You can set a default mount position here if desired:
        # for f in spec.worldbody.frames:
        #     if f.name.startswith(self._mount_prefix):
        #         f.pos = (0.45, -0.03, 0.90)
        #         f.quat = (1, 0, 0, 0)
