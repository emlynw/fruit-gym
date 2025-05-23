#!/usr/bin/env python3
"""
resize_textures_mixed.py
------------------------
• 512-px resize for normal textures
• 1024-px resize for   textures/skyboxes/**

Run from the directory that holds  “textures/”, or pass --src / --dst.
"""

import argparse, pathlib
from PIL import Image

def target_width(rel_path: pathlib.Path, default_w: int, skybox_w: int) -> int:
    """Return the width you want for this particular file."""
    return skybox_w if "skyboxes" in rel_path.parts else default_w

def resize_and_save(src: pathlib.Path, dst: pathlib.Path, width_out: int):
    with Image.open(src) as im:
        w, h = im.size
        if w <= width_out:         # already small enough
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst)
            return
        new_h = round(h * width_out / w)
        im = im.resize((width_out, new_h), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src",   default="textures")
    p.add_argument("--dst",   default="textures_small")
    p.add_argument("--std",   type=int, default=512,
                   help="width for regular textures (default 512)")
    p.add_argument("--sky",   type=int, default=1024,
                   help="width for skybox textures (default 1024)")
    args = p.parse_args()

    src_root, dst_root = map(pathlib.Path, (args.src, args.dst))
    exts = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".exr"}

    files = [p for p in src_root.rglob("*") if p.suffix.lower() in exts]
    for i, src_path in enumerate(files, 1):
        rel = src_path.relative_to(src_root)
        w_out = target_width(rel, args.std, args.sky)
        dst_path = dst_root / rel
        resize_and_save(src_path, dst_path, w_out)
        print(f"[{i}/{len(files)}] {rel}  →  width {w_out}")

if __name__ == "__main__":
    main()
