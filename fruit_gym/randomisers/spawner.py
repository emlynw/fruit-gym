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
    """Attach *N* copies of *strawb_xml* under unique frames in the scene."""

    affects_spec = True
    needs_ctx = False  # does not need a MjrContext

    def __init__(
        self,
        strawb_xml: Path | str = "strawb.xml",
        min_count: int = 4,
        max_count: int = 8,
        mount_prefix: str = "vine_",
    ):
        self._strawb_xml = Path(strawb_xml)
        self._min = min_count
        self._max = max_count
        self._mount_prefix = mount_prefix
        if not self._strawb_xml.exists():
            raise FileNotFoundError(self._strawb_xml)

    def apply(self, *, spec, model, data, rng, ctx=None):
        if spec is None:
            raise ValueError("StrawberrySpawnerRandomiser requires spec when affects_spec=True")
        
        # clear previous mounts (in case of reset w/o new base spec)
        spec.worldbody.frames.clear()  
        count = rng.integers(self._min, self._max + 1)
        first = True
        for i in range(count):
            vine_spec = mujoco.MjSpec.from_file(str(self._strawb_xml))
            if first:
                _move_unique_assets(spec, vine_spec)
                first = False
            frame = spec.worldbody.add_frame(name=f"{self._mount_prefix}{i}")
            frame.attach_body(vine_spec.worldbody.bodies[0], suffix=f"_{i}")