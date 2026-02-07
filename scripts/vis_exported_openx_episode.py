#!/usr/bin/env python3
"""
vis_exported_openx_episode.py

Visualize an episode previously exported by save_openx_episode.py into "openx_exports".

Expected folder structure (from the saver):
  <out_root>/<dataset>/<split>/episode_<idx>/
    meta.json
    joints.npz
    images/rgb_000000.png (or .npy if you used --save_npy)
    images_wrist/wrist_000000.png (or .npy)

Usage:
  python vis_exported_openx_episode.py \
    --root openx_exports --dataset jaco_play --split train --episode 0 --steps 12

  # if you exported images as .npy:
  python vis_exported_openx_episode.py \
    --root openx_exports --dataset jaco_play --split train --episode 0 --steps 12 --npy

Options:
  --stride K   visualize every K-th saved frame (default 1)
  --show_joints   show joints values under the RGB image
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def _read_image(path: Path, use_npy: bool = False) -> np.ndarray:
    """Read PNG/JPG via TF or NPY; returns uint8 HxWxC."""
    if use_npy or path.suffix.lower() == ".npy":
        img = np.load(path)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = img[..., None]
        return img

    b = path.read_bytes()
    t = tf.io.decode_image(b, channels=3, expand_animations=False)  # uint8
    return t.numpy()


def _summarize_joints(j: np.ndarray, valid: Optional[np.ndarray] = None, max_items: int = 7) -> str:
    if j.size == 0:
        return "joints: (missing)"
    if valid is not None and valid.size == j.size:
        j = j[valid.astype(bool)]
    flat = j.reshape(-1)
    shown = np.round(flat[:max_items], 3)
    more = "" if flat.size <= max_items else f" …(+{flat.size - max_items})"
    return f"joints: {shown}{more}"


def load_episode_dir(root: Path, dataset: str, split: str, episode: int) -> Path:
    ep_dir = root / dataset / split / f"episode_{episode:06d}"
    if not ep_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {ep_dir}")
    if not (ep_dir / "meta.json").exists():
        raise FileNotFoundError(f"meta.json not found in: {ep_dir}")
    return ep_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="openx_exports", help="Export root folder")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name folder (e.g., jaco_play)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12, help="How many frames to show")
    parser.add_argument("--stride", type=int, default=1, help="Show every stride-th saved frame")
    parser.add_argument("--npy", action="store_true", help="Load images from .npy (if you exported with --save_npy)")
    parser.add_argument("--show_joints", action="store_true", help="Show joints text under RGB images")
    parser.add_argument("--no_wrist", action="store_true", help="Do not show wrist row")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    ep_dir = load_episode_dir(root, args.dataset, args.split, args.episode)

    meta = json.loads((ep_dir / "meta.json").read_text())
    joints_path = ep_dir / "joints.npz"
    if not joints_path.exists():
        raise FileNotFoundError(f"Missing joints.npz: {joints_path}")

    joints_npz = np.load(joints_path)
    joints = joints_npz["joints"]            # [N, D]
    joints_valid = joints_npz["joints_valid"]  # [N, D] uint8
    step_index = joints_npz["step_index"]    # [N]
    timestamp = joints_npz["timestamp"]      # [N]
    print('joints', joints.shape)

    rgb_files: List[str] = meta.get("rgb_files", [])
    wrist_files: List[str] = meta.get("wrist_files", [])

    if len(rgb_files) == 0:
        raise ValueError("meta.json has empty rgb_files.")

    # Determine how many frames are available
    N = len(rgb_files)
    if joints.shape[0] != N:
        print(f"[WARN] joints has {joints.shape[0]} rows but meta has {N} rgb files. Will use min().")
    N_use = min(N, joints.shape[0])

    stride = max(1, args.stride)
    idxs = list(range(0, N_use, stride))
    if args.steps > 0:
        idxs = idxs[: args.steps]
    T = len(idxs)
    if T == 0:
        raise ValueError("No frames to visualize (check --steps/--stride).")

    instr = meta.get("instruction", None)
    title_lines = [f"{args.dataset}/{args.split}/episode_{args.episode:06d}  (frames shown: {T}/{N_use}, stride={stride})"]
    if instr:
        title_lines.append(f"Instruction: {instr}")

    rows = 1 if args.no_wrist else 2
    fig, axes = plt.subplots(rows, T, figsize=(2.3 * T, 3.3 * rows))
    if T == 1:
        axes = np.expand_dims(axes, axis=1)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for col, i in enumerate(idxs):
        rgb_rel = rgb_files[i]
        rgb_path = ep_dir / rgb_rel
        rgb = _read_image(rgb_path, use_npy=args.npy)

        ax0 = axes[0, col]
        ax0.imshow(rgb)

        # Title: step and timestamp
        ts = timestamp[i]
        st = int(step_index[i]) if i < len(step_index) else i
        ax0.set_title(f"frame {i}\nstep {st}\n{ts:.3f}s" if np.isfinite(ts) else f"frame {i}\nstep {st}", fontsize=8)

        if args.show_joints:
            j = joints[i]
            v = joints_valid[i] if joints_valid.shape[0] > i else None
            ax0.set_xlabel(_summarize_joints(j, v), fontsize=7)

        if not args.no_wrist:
            wrist_rel = wrist_files[i] if i < len(wrist_files) else ""
            ax1 = axes[1, col]
            if wrist_rel:
                wrist_path = ep_dir / wrist_rel
                if wrist_path.exists():
                    wrist = _read_image(wrist_path, use_npy=args.npy)
                    ax1.imshow(wrist)
                else:
                    ax1.text(0.5, 0.5, "wrist missing", ha="center", va="center")
            else:
                ax1.text(0.5, 0.5, "wrist missing", ha="center", va="center")
            ax1.axis("off")
            if col == 0:
                ax1.set_title("wrist", fontsize=9)

    fig.suptitle("\n".join(title_lines), fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
