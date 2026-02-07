#!/usr/bin/env python3
"""
save_openx_episode.py

Save one Open X-Embodiment RLDS/TFDS episode to a folder:
- RGB images (observation[image_key])
- Wrist images (observation[wrist_key] or observation["wrist_image"])
- Robot joints at each saved frame (defaults to observation["joint_pos"] if present)
- A metadata JSON + a joints.npz for easy loading

This script mirrors the loading/decoding logic from your visualizer.

Example:
  python save_openx_episode.py \
    --dataset_dir "D:/openx/jaco_play/0.1.0" --split train --episode 0 \
    --out_dir "D:/openx_exports" \
    --stride 1 --image_key image --wrist_key image_wrist --joint_key joint_pos
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# -----------------------------
# Helpers: bytes / image / text
# -----------------------------

def _maybe_decode_scalar_bytes(x: Any) -> Optional[bytes]:
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, np.ndarray) and x.shape == ():
        v = x.item()
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
    return None


def decode_text_maybe(x: Any) -> Optional[str]:
    if x is None:
        return None
    b = _maybe_decode_scalar_bytes(x)
    if b is not None:
        return b.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        return x
    if isinstance(x, np.ndarray) and x.shape == ():
        v = x.item()
        if isinstance(v, str):
            return v
    return None


def decode_image_maybe(x: Any, channels: int = 3) -> np.ndarray:
    """
    Decode image from:
      - encoded bytes (jpeg/png/etc), or
      - numpy array (HxW or HxWxC).
    Returns uint8 numpy array.
    """
    # Already image array
    if isinstance(x, np.ndarray) and x.ndim in (2, 3):
        img = x
        if img.dtype != np.uint8:
            mx = float(np.nanmax(img)) if img.size else 0.0
            if mx <= 1.5:
                img = np.clip(img * 255.0, 0, 255)
            else:
                img = np.clip(img, 0, 255)
            img = img.astype(np.uint8)
        # If grayscale, keep as HxW; caller can handle
        return img

    # Encoded bytes
    b = _maybe_decode_scalar_bytes(x)
    if b is not None:
        t = tf.io.decode_image(b, channels=channels, expand_animations=False)  # uint8
        return t.numpy()

    raise TypeError(
        f"Don't know how to decode image: type={type(x)}, "
        f"shape={getattr(x,'shape',None)}, dtype={getattr(x,'dtype',None)}"
    )


# -----------------------------
# TFDS / RLDS loading utilities
# -----------------------------

def get_episode(builder: tfds.core.DatasetBuilder, split: str, episode_index: int) -> dict:
    ds = builder.as_dataset(split=split, shuffle_files=False)
    for i, ep in enumerate(tfds.as_numpy(ds)):
        if i == episode_index:
            return ep
    raise ValueError(f"Episode index {episode_index} not found in split={split}.")


def list_steps(ep: dict) -> List[dict]:
    steps = list(ep["steps"])
    if not steps:
        raise ValueError("Episode has zero steps.")
    return steps


# -----------------------------
# Saving utilities
# -----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_png(path: Path, img_u8: np.ndarray) -> None:
    """
    Write uint8 image as PNG using TF (avoids PIL/cv2 dependency).
    Supports HxW (grayscale) or HxWxC (C=1/3/4).
    """
    if img_u8.dtype != np.uint8:
        raise ValueError(f"Expected uint8, got {img_u8.dtype}")
    t = tf.convert_to_tensor(img_u8)
    if t.ndim == 2:
        t = t[..., None]  # HxWx1
    encoded = tf.io.encode_png(t).numpy()
    path.write_bytes(encoded)


def _get_obs_field(obs: dict, keys: List[str]) -> Tuple[Optional[Any], Optional[str]]:
    """Return (value, key_used) for the first key in keys that exists."""
    for k in keys:
        if k in obs and obs[k] is not None:
            return obs[k], k
    return None, None


def _as_float_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.asarray(x)
        if arr.dtype == object:
            # Sometimes object arrays hold scalars/arrays; best effort
            arr = np.array(arr.tolist())
        return arr.astype(np.float32)
    except Exception:
        return None


def save_episode(
    builder: tfds.core.DatasetBuilder,
    split: str,
    episode_index: int,
    out_root: Path,
    image_key: str = "image",
    wrist_key: str = "image_wrist",
    joint_key: str = "joint_pos",
    stride: int = 1,
    channels: int = 3,
    save_png: bool = True,
) -> Path:
    """
    Save episode to:
      out_root/<dataset_name>/<split>/episode_<idx>/
        images/rgb_000000.png
        images_wrist/wrist_000000.png
        joints.npy or joints.npz
        meta.json

    Returns the episode directory.
    """
    ep = get_episode(builder, split, episode_index)
    steps = list_steps(ep)

    # Try to infer dataset name from builder
    ds_name = getattr(builder.info, "name", "openx_dataset")

    ep_dir = out_root / ds_name / split / f"episode_{episode_index:06d}"
    img_dir = ep_dir / "images"
    wrist_dir = ep_dir / "images_wrist"
    _ensure_dir(img_dir)
    _ensure_dir(wrist_dir)

    # Episode-level instruction (best-effort)
    instr = None
    try:
        obs0 = steps[0].get("observation", {})
        instr = decode_text_maybe(obs0.get("natural_language_instruction", None))
        if instr is None:
            instr = decode_text_maybe(steps[0].get("language_instruction", None))
    except Exception:
        instr = None

    joints_list: List[np.ndarray] = []
    step_indices: List[int] = []
    img_files: List[str] = []
    wrist_files: List[str] = []

    # Determine which wrist key is available (per-step can vary, so do best-effort each step)
    wrist_fallback_keys = [wrist_key, "wrist_image", "image_wrist"]

    # Optional: record some timestamps if present
    timestamps: List[float] = []

    for t in range(0, len(steps), max(1, stride)):
        s = steps[t]
        obs = s.get("observation", {})

        # Images
        rgb_raw, rgb_used = _get_obs_field(obs, [image_key, "image"])
        wrist_raw, wrist_used = _get_obs_field(obs, wrist_fallback_keys)

        if rgb_raw is None:
            raise KeyError(f"Step {t}: no RGB image found under keys {[image_key, 'image']}")

        rgb = decode_image_maybe(rgb_raw, channels=channels)
        wrist = None
        if wrist_raw is not None:
            wrist = decode_image_maybe(wrist_raw, channels=channels)

        # Joints
        joint_raw, joint_used = _get_obs_field(obs, [joint_key, "joint_pos", "joints", "robot_joint_positions"])
        joints = _as_float_array(joint_raw)

        # Timestamp (best-effort; RLDS often has 'timestamp' fields, varies by dataset)
        ts = None
        for k in ["timestamp", "t", "time", "step_timestamp"]:
            if k in s:
                try:
                    ts = float(np.asarray(s[k]).reshape(()))
                    break
                except Exception:
                    pass
            if k in obs:
                try:
                    ts = float(np.asarray(obs[k]).reshape(()))
                    break
                except Exception:
                    pass
        timestamps.append(ts if ts is not None else float("nan"))

        # Write images
        frame_id = len(step_indices)
        rgb_name = f"rgb_{frame_id:06d}.png"
        wrist_name = f"wrist_{frame_id:06d}.png"

        if save_png:
            _write_png(img_dir / rgb_name, rgb)
            img_files.append(str(Path("images") / rgb_name))
            if wrist is not None:
                _write_png(wrist_dir / wrist_name, wrist)
                wrist_files.append(str(Path("images_wrist") / wrist_name))
            else:
                wrist_files.append("")  # keep alignment
        else:
            # If you prefer raw npy images:
            np.save(img_dir / f"rgb_{frame_id:06d}.npy", rgb)
            img_files.append(str(Path("images") / f"rgb_{frame_id:06d}.npy"))
            if wrist is not None:
                np.save(wrist_dir / f"wrist_{frame_id:06d}.npy", wrist)
                wrist_files.append(str(Path("images_wrist") / f"wrist_{frame_id:06d}.npy"))
            else:
                wrist_files.append("")

        step_indices.append(t)

        if joints is None:
            # Keep placeholder so arrays stay aligned with frames
            joints_list.append(np.array([], dtype=np.float32))
        else:
            joints_list.append(joints.reshape(-1).astype(np.float32))

    # Stack joints into a ragged-friendly object if lengths vary
    # (Most robot datasets have fixed DOF, but be robust.)
    max_len = max((j.shape[0] for j in joints_list), default=0)
    if max_len == 0:
        joints_mat = np.zeros((len(joints_list), 0), dtype=np.float32)
        joints_mask = np.zeros((len(joints_list), 0), dtype=np.uint8)
    else:
        joints_mat = np.zeros((len(joints_list), max_len), dtype=np.float32)
        joints_mask = np.zeros((len(joints_list), max_len), dtype=np.uint8)
        for i, j in enumerate(joints_list):
            n = min(max_len, j.shape[0])
            if n > 0:
                joints_mat[i, :n] = j[:n]
                joints_mask[i, :n] = 1

    np.savez_compressed(
        ep_dir / "joints.npz",
        joints=joints_mat,
        joints_valid=joints_mask,
        step_index=np.asarray(step_indices, dtype=np.int32),
        timestamp=np.asarray(timestamps, dtype=np.float64),
    )

    meta = {
        "dataset": ds_name,
        "split": split,
        "episode_index": int(episode_index),
        "num_steps_total": int(len(steps)),
        "stride": int(stride),
        "num_frames_saved": int(len(step_indices)),
        "image_key_requested": image_key,
        "wrist_key_requested": wrist_key,
        "joint_key_requested": joint_key,
        "instruction": instr,
        "rgb_files": img_files,
        "wrist_files": wrist_files,
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[OK] Saved episode to: {ep_dir}")
    print(f"     Frames saved: {len(step_indices)} (stride={stride})")
    print(f"     RGB dir:      {img_dir}")
    print(f"     Wrist dir:    {wrist_dir}")
    print(f"     Joints:       {ep_dir / 'joints.npz'}")
    print(f"     Meta:         {ep_dir / 'meta.json'}")

    return ep_dir


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="tensorflow_datasets/jaco_play/0.1.0",
        help="Path to downloaded dataset directory, e.g. D:/openx/jaco_play/0.1.0",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="openx_exports",
        help="Output root directory to save episodes",
    )
    parser.add_argument("--stride", type=int, default=1, help="Save every stride-th step")
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--wrist_key", type=str, default="image_wrist")
    parser.add_argument("--joint_key", type=str, default="joint_pos")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--save_npy", action="store_true", help="Save images as .npy instead of .png")
    args = parser.parse_args()

    dataset_dir = str(Path(args.dataset_dir).expanduser())
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(dataset_dir)

    out_root = Path(args.out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    # If TF GPU causes issues on Windows, uncomment:
    # tf.config.set_visible_devices([], "GPU")

    builder = tfds.builder_from_directory(dataset_dir)
    builder.download_and_prepare(download_dir=dataset_dir)  # safe no-op if already prepared

    save_episode(
        builder=builder,
        split=args.split,
        episode_index=args.episode,
        out_root=out_root,
        image_key=args.image_key,
        wrist_key=args.wrist_key,
        joint_key=args.joint_key,
        stride=max(1, args.stride),
        channels=args.channels,
        save_png=not args.save_npy,
    )


if __name__ == "__main__":
    main()
