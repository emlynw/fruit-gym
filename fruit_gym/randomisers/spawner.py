from .base import Randomiser
from pathlib import Path
import mujoco

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

class SpawnerRandomiser(Randomiser):
    """Attach N copies of a randomly chosen plant XML under unique frames.

    Accepts either:
      - strawb_xml=Path|str (legacy single-file mode), or
      - xml_choices=[Path|str, ...] (random choice per spawn).
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

        self._min = min_count
        self._max = max_count
        self._mount_prefix = mount_prefix

    def apply(self, *, spec, model, data, rng, ctx=None):
        if spec is None:
            raise ValueError("SpawnerRandomiser needs a spec when affects_spec=True")

        # clear previous mounts (in case of reset w/o new base spec)
        spec.worldbody.frames.clear()
        count = rng.integers(self._min, self._max + 1)

        for i in range(count):
            choice_path = Path(rng.choice(self._choices))
            vine_spec = mujoco.MjSpec.from_file(str(choice_path))

            _move_unique_assets(spec, vine_spec)

            frame = spec.worldbody.add_frame(name=f"{self._mount_prefix}{i}")
            # Tag leaves so later passes can spot them if needed
            suffix = f"_leaves_{i}" if "leaves" in choice_path.stem.lower() else f"_{i}"
            frame.attach_body(vine_spec.worldbody.bodies[0], suffix=suffix)
