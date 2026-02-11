#!/usr/bin/env python3
"""
Create binary masks per category from COCO polygon segmentations.

Output:
  out_dir/
    <image_stem>__cat-<id>-<name>.png   (single-channel 0/255)
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw


def build_image_path(images_dir: str, file_name: str) -> str:
    # COCO file_name may contain subfolders; join safely
    file_name = file_name.lstrip("/\\")
    return os.path.normpath(os.path.join(images_dir, file_name))


def sanitize(s: str) -> str:
    # safe filename chunk
    keep = []
    for ch in (s or ""):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out if out else "cat"


def parse_coco_polygons(seg):
    """
    COCO polygon segmentation:
      - list of lists: [[x1,y1,x2,y2,...], [...], ...]
      - or flat list for single polygon: [x1,y1,...]
    Returns list of polygons, each as list of (x,y) float tuples.
    Ignores RLE.
    """
    if seg is None:
        return []
    if not isinstance(seg, list):
        return []  # RLE/dict etc.

    if len(seg) == 0:
        return []

    # flat list -> wrap
    if isinstance(seg[0], (int, float)):
        seg = [seg]

    polys = []
    for p in seg:
        if not isinstance(p, list) or len(p) < 6:
            continue
        coords = np.asarray(p, dtype=np.float32).reshape(-1, 2)
        polys.append([(float(x), float(y)) for x, y in coords])
    return polys


def rasterize_category_mask(width: int, height: int, polygons):
    """
    polygons: list of polygons; each polygon is list of (x,y).
    returns uint8 mask in {0,255} shape (H,W)
    """
    mask_img = Image.new("L", (width, height), 0)  # 0 background
    draw = ImageDraw.Draw(mask_img)

    # Fill each polygon with 255
    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        draw.polygon(poly, outline=255, fill=255)

    return np.array(mask_img, dtype=np.uint8)


def make_masks_per_category(coco_json: str, images_dir: str, out_dir: str, use_coco_size: bool = True):
    os.makedirs(out_dir, exist_ok=True)

    with open(coco_json, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    cat_id_to_name = {c.get("id"): c.get("name", str(c.get("id"))) for c in cats}

    # Group annotations by image_id
    anns_by_image = defaultdict(list)
    for a in anns:
        if "image_id" in a:
            anns_by_image[a["image_id"]].append(a)

    images_sorted = sorted(images, key=lambda x: x.get("id", 0))

    for im in images_sorted:
        image_id = im.get("id")
        file_name = im.get("file_name", "")
        if image_id is None or not file_name:
            print(f"[WARN] Skipping image with missing id/file_name: {im}")
            continue

        # Determine mask size
        if use_coco_size and im.get("width") and im.get("height"):
            width, height = int(im["width"]), int(im["height"])
        else:
            # fallback: read the image to get size
            img_path = build_image_path(images_dir, file_name)
            if not os.path.exists(img_path):
                print(f"[WARN] Image not found for size fallback: {img_path}; skipping.")
                continue
            with Image.open(img_path) as pil_img:
                width, height = pil_img.size

        # Collect polygons per category for this image
        polys_by_cat = defaultdict(list)
        for ann in anns_by_image.get(image_id, []):
            cat_id = ann.get("category_id")
            if cat_id is None:
                continue
            seg = ann.get("segmentation")
            polys = parse_coco_polygons(seg)
            if polys:
                polys_by_cat[cat_id].extend(polys)

        if not polys_by_cat:
            # no masks for this image
            continue

        image_stem = os.path.splitext(os.path.basename(file_name))[0]

        for cat_id, polygons in polys_by_cat.items():
            mask = rasterize_category_mask(width, height, polygons)

            cat_name = sanitize(cat_id_to_name.get(cat_id, str(cat_id)))
            out_name = f"{image_stem}__cat-{cat_id}-{cat_name}.png"
            out_path = os.path.join(out_dir, out_name)

            Image.fromarray(mask, mode="L").save(out_path)

        print(f"[OK] {file_name}: wrote {len(polys_by_cat)} category mask(s)")


def make_args():
    p = argparse.ArgumentParser(description="Create binary masks per category from COCO polygon segmentations.")
    p.add_argument("--coco_json", type=str, required=True, help="Path to COCO-format JSON.")
    p.add_argument("--images_dir", type=str, required=True, help="Directory that contains the images (for fallback sizing).")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for masks.")
    p.add_argument(
        "--use_coco_size",
        action="store_true",
        help="Use width/height from COCO 'images' entries (recommended).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = make_args()
    make_masks_per_category(
        coco_json=args.coco_json,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        use_coco_size=args.use_coco_size,
    )
