#!/usr/bin/env python3
"""
Visualize COCO-format instance segmentations (polygons) by overlaying them on images.

Behavior:
- Shows images one-by-one in an OpenCV window.
- When you close the window, it automatically advances to the next image.

Notes:
- Supports polygon segmentations (COCO "segmentation" as list-of-lists).
- Also draws bounding boxes and category names (if present).
"""

import os
import json
import argparse
from collections import defaultdict

import cv2
import numpy as np


def _safe_imread(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def _build_image_path(images_dir: str, file_name: str):
    # COCO "file_name" can include subfolders; join safely
    file_name = file_name.lstrip("/\\")
    return os.path.normpath(os.path.join(images_dir, file_name))


def _color_for_id(idx: int):
    # Deterministic-ish BGR color from an integer id
    rng = np.random.default_rng(seed=int(idx) & 0xFFFFFFFF)
    c = rng.integers(40, 255, size=3, dtype=np.int32)
    return (int(c[0]), int(c[1]), int(c[2]))  # B, G, R


def _draw_polygons_overlay(img, polygons, color_bgr, alpha=0.35):
    """
    polygons: list of polygons; each polygon is Nx2 float coords
    """
    if not polygons:
        return img

    overlay = img.copy()
    h, w = img.shape[:2]

    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue

        pts = np.asarray(poly, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue

        # Clip to image bounds
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        pts_i = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_i], color_bgr)
        cv2.polylines(img, [pts_i], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return out


def _parse_coco_polygons(seg):
    """
    COCO polygon segmentation formats:
      - seg = [ [x1,y1,x2,y2,...], [ ... ], ... ]  (multi-poly)
    We ignore RLE here.
    Returns: list of polygons, each polygon is Nx2
    """
    if seg is None:
        return []

    # Polygon case: list of lists (or list of numbers for single polygon)
    if isinstance(seg, list):
        polys = []
        # If it's a flat list of numbers, wrap it
        if len(seg) > 0 and isinstance(seg[0], (int, float)):
            seg = [seg]

        for p in seg:
            if not isinstance(p, list) or len(p) < 6:
                continue
            coords = np.asarray(p, dtype=np.float32).reshape(-1, 2)
            polys.append(coords)
        return polys

    # RLE/dict case -> skip
    return []


def visualize_coco(coco_json: str, images_dir: str, window_name: str = "COCO Viewer", start_index: int = 0):
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

    if not images:
        raise ValueError("No 'images' found in COCO JSON.")

    # Sort images by id for deterministic traversal
    images_sorted = sorted(images, key=lambda x: x.get("id", 0))

    # OpenCV: allow closing the window to proceed
    for i, iminfo in enumerate(images_sorted[start_index:], start=start_index):
        image_id = iminfo.get("id")
        file_name = iminfo.get("file_name")
        if file_name is None:
            print(f"[WARN] image entry missing file_name (image_id={image_id}); skipping.")
            continue

        img_path = _build_image_path(images_dir, file_name)
        img = _safe_imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}; skipping.")
            continue

        draw = img.copy()
        these_anns = anns_by_image.get(image_id, [])

        # Draw each annotation
        for ann in these_anns:
            seg = ann.get("segmentation")
            bbox = ann.get("bbox")  # [x,y,w,h]
            cat_id = ann.get("category_id")
            ann_id = ann.get("id", 0)

            color = _color_for_id(cat_id if cat_id is not None else ann_id)

            polys = _parse_coco_polygons(seg)
            if polys:
                draw = _draw_polygons_overlay(draw, polys, color_bgr=color, alpha=0.35)

            # Draw bbox + label if available
            if isinstance(bbox, list) and len(bbox) == 4:
                x, y, w, h = bbox
                x1, y1 = int(round(x)), int(round(y))
                x2, y2 = int(round(x + w)), int(round(y + h))
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

                label = cat_id_to_name.get(cat_id, str(cat_id)) if cat_id is not None else "obj"
                cv2.putText(
                    draw,
                    label,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )

        title = f"{window_name}  [{i+1}/{len(images_sorted)}]  id={image_id}  {os.path.basename(file_name)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, draw)
        cv2.setWindowTitle(window_name, title)

        # Wait until window is closed by the user
        while True:
            # getWindowProperty returns < 1 when closed
            prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            if prop < 1:
                # window closed -> proceed to next image
                break
            # small wait to keep UI responsive
            key = cv2.waitKey(30)
            # Optional: allow skipping with ESC/Q (doesn't close window; just advances)
            if key in (27, ord("q"), ord("Q")):
                cv2.destroyWindow(window_name)
                break

    # Clean up
    cv2.destroyAllWindows()


def make_args():
    p = argparse.ArgumentParser(description="Visualize COCO polygon segmentations overlayed on images.")
    p.add_argument("--coco_json", type=str, required=True, help="Path to COCO-format JSON file.")
    p.add_argument("--images_dir", type=str, required=True, help="Directory containing the images referenced by file_name.")
    p.add_argument("--start_index", type=int, default=0, help="Start from the N-th image in sorted order.")
    p.add_argument("--window_name", type=str, default="COCO Viewer", help="OpenCV window name.")
    return p.parse_args()


if __name__ == "__main__":
    args = make_args()
    visualize_coco(
        coco_json=args.coco_json,
        images_dir=args.images_dir,
        window_name=args.window_name,
        start_index=args.start_index,
    )
