#!/usr/bin/env python3
"""
Resize all images in a folder to the smallest resolution among them.

Usage:
    python resize_to_smallest.py /path/to/folder
    python resize_to_smallest.py /path/to/folder -o /path/to/output
    python resize_to_smallest.py /path/to/folder --overwrite
"""

import argparse
from pathlib import Path

from PIL import Image, ImageOps   # pip install pillow


# ---------- helpers --------------------------------------------------------- #
def find_smallest_resolution(image_paths):
    """Return (Path, (w, h)) of the image with the smallest total pixels."""
    smallest_file = None
    smallest_size = None  # (w, h)

    for path in image_paths:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)  # respect camera orientation
            size = im.size  # (w, h)
        if smallest_size is None or (size[0] * size[1] < smallest_size[0] * smallest_size[1]):
            smallest_file, smallest_size = path, size

    return smallest_file, smallest_size


def resize_images(image_paths, target_size, output_dir=None, overwrite=False, keep_format=True):
    """Resize all images to *target_size* and save them."""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            resized = im.resize(target_size, Image.LANCZOS)

            # Decide where to write the file
            if output_dir:
                out_path = output_dir / path.name
            else:
                out_path = path if overwrite else path.with_stem(f"{path.stem}_{target_size[0]}x{target_size[1]}")

            save_kwargs = {"format": im.format} if keep_format else {}  # default: preserve original format
            resized.save(out_path, **save_kwargs)
            print(f"✓ {out_path}")


# ---------- main ------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Resize every image in a folder to the smallest resolution found.")
    parser.add_argument("folder", type=Path, help="Folder containing images.")
    parser.add_argument("-o", "--output", type=Path, help="Optional output directory. If omitted, files are written next to originals.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite originals (ignored if --output is given).")
    parser.add_argument("--no-keep-format", action="store_true", help="Save all output images as PNG.")
    args = parser.parse_args()

    # Collect candidate files (add or remove suffixes as needed)
    img_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    images = [p for p in args.folder.iterdir() if p.suffix.lower() in img_suffixes]

    if not images:
        print("No images found in", args.folder)
        return

    smallest_path, smallest_size = find_smallest_resolution(images)
    print(f"Smallest image: {smallest_path.name} → {smallest_size[0]}×{smallest_size[1]}")

    resize_images(
        images,
        smallest_size,
        output_dir=args.output,
        overwrite=args.overwrite,
        keep_format=not args.no_keep_format,
    )

    print("All done!")


if __name__ == "__main__":
    main()