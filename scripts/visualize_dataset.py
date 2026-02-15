#!/usr/bin/env python3
"""
visualize_dataset.py

Robust visualizer for Open X-Embodiment jaco_play downloaded locally as an RLDS/TFDS directory dataset.

- Loads builder via tfds.builder_from_directory(<local_path>)
- Grabs an episode by index
- Decodes observation["image"] and observation["image_wrist"] (bytes or arrays)
- Displays first N steps
- Robustly summarizes action even if it's a nested dict

Usage:
  python visualize_dataset.py --dataset_dir "D:/openx/jaco_play/0.1.0" --split train --episode 0 --steps 12
"""

import argparse
import os
from pathlib import Path
from typing import Any, Optional, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


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
# Helpers: action summarization
# -----------------------------

def _is_numeric_array(a: np.ndarray) -> bool:
    return isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.number)


def _flatten_numeric(x: Any) -> Optional[np.ndarray]:
    """Try to convert x to a numeric numpy array; return None if not possible."""
    try:
        arr = np.asarray(x)
        if _is_numeric_array(arr):
            return arr
        return None
    except Exception:
        return None


def summarize_action(action: Any, max_items: int = 3) -> str:
    """
    Return a short string summary of action that works for:
      - numeric arrays
      - dicts (possibly nested)
      - other objects
    """
    # Numeric array / list
    arr = _flatten_numeric(action)
    if arr is not None:
        flat = arr.reshape(-1)
        shown = np.round(flat[:max_items], 2)
        more = "" if flat.size <= max_items else f" …(+{flat.size - max_items})"
        return f"{shown}"

    # Dict (possibly nested)
    if isinstance(action, dict):
        parts = []
        for k, v in action.items():
            arr_v = _flatten_numeric(v)
            if arr_v is not None:
                flat = arr_v.reshape(-1)
                shown = np.round(flat[:max_items], 2)
                more = "" if flat.size <= max_items else f"…(+{flat.size - max_items})"
                parts.append(f"{k}:{shown}{more}\n")
            else:
                # If nested dict, just show keys
                if isinstance(v, dict):
                    parts.append(f"{k}:(dict keys={list(v.keys())[:4]})\n")
                else:
                    parts.append(f"{k}:(type {type(v).__name__})\n")
        return "action{" + ", ".join(parts) + "}"

    # Fallback
    return f"action(type={type(action).__name__})"


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


def print_structure(ep: dict, steps: list) -> None:
    print("Episode keys:", ep.keys())
    print("Step keys:", steps[0].keys())
    if "observation" in steps[0] and isinstance(steps[0]["observation"], dict):
        print("Observation keys:", steps[0]["observation"].keys())


# -----------------------------
# Visualization
# -----------------------------

def show_episode_images(
    steps: List[dict],
    n_steps: int = 12,
    image_key: str = "image",
    wrist_key: str = "image_wrist",
    show_actions: bool = True,
    show_instr: bool = True,
) -> None:
    T = min(n_steps, len(steps))
    if T <= 0:
        raise ValueError("n_steps must be >= 1")

    instr = None
    if show_instr:
        instr = decode_text_maybe(steps[0]["observation"].get("natural_language_instruction", None))
        if instr is None:
            instr = decode_text_maybe(steps[0].get("language_instruction", None))

    fig, axes = plt.subplots(2, T, figsize=(2.3 * T, 5.0))
    if T == 1:
        axes = np.expand_dims(axes, axis=1)

    for t in range(T):
        s = steps[t]
        obs = s["observation"]

        img = decode_image_maybe(obs[image_key], channels=3)

        im = obs.get(wrist_key, None)
        if im is None:
            im = obs.get("wrist_image", None)
        wrist = decode_image_maybe(im, channels=3)

        ax0 = axes[0, t]
        ax1 = axes[1, t]

        ax0.imshow(img)
        #ax0.axis("off")
        if t == 0:
            ax0.set_title(image_key, fontsize=10)

        ax1.imshow(wrist)
        ax1.axis("off")
        if t == 0:
            ax1.set_title(wrist_key, fontsize=10)

        if show_actions and "action" in s:
            ax0.set_xlabel(summarize_action(s["action"]), fontsize=7)

    title_lines = [f"OpenX episode visualization ({T} steps shown)"]
    if instr:
        title_lines.append(f"Instruction: {instr}")
    fig.suptitle("\n".join(title_lines), fontsize=11)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="tensorflow_datasets/jaco_play/0.1.0",
                        help="Path to downloaded dataset directory, e.g. D:/openx/jaco_play/0.1.0")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--wrist_key", type=str, default="image_wrist")
    parser.add_argument("--no_actions", action="store_true")
    parser.add_argument("--no_instr", action="store_true")
    args = parser.parse_args()

    dataset_dir = str(Path(args.dataset_dir).expanduser())
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(dataset_dir)

    # If TF GPU causes issues on Windows, uncomment:
    # tf.config.set_visible_devices([], "GPU")

    builder = tfds.builder_from_directory(dataset_dir)
    builder.download_and_prepare(download_dir=dataset_dir)  # safe no-op if already prepared

    ep = get_episode(builder, args.split, args.episode)
    steps = list_steps(ep)

    print_structure(ep, steps)
    print(f"Episode {args.episode}: {len(steps)} steps in split='{args.split}'")

    # Print types for quick debugging
    obs0 = steps[0]["observation"]
    print("observation keys:", steps[0]["observation"].keys())
    
    for k in [args.image_key, args.wrist_key]:
        x = obs0.get(k, None)
        print(f"{k}: type={type(x)} shape={getattr(x, 'shape', None)} dtype={getattr(x, 'dtype', None)}")

    if 'joint_pos' in steps[0]["observation"].keys():
        jp = steps[0]["observation"]["joint_pos"]
        print("joint_pos:", np.asarray(jp).shape, np.asarray(jp)[:10])

    if 'nd_effector_cartesian_pos' in steps[0]["observation"].keys():
        ee = steps[0]["observation"]["end_effector_cartesian_pos"]
        print("ee_pos:", np.asarray(ee).shape, np.asarray(ee))   


    if "action" in steps[0]:
        print('==============================================')
        print("action type:", type(steps[0]["action"]))
        if isinstance(steps[0]["action"], dict):
            print("action keys:", steps[0]["action"].keys())

            for i in range(len(steps)):
                print(i, steps[i]["action"]['world_vector'])
        else:
            print(steps[0]["action"])
        print('==============================================')                    

    show_episode_images(
        steps[::10],
        n_steps=args.steps,
        image_key=args.image_key,
        wrist_key=args.wrist_key,
        show_actions=not args.no_actions,
        show_instr=not args.no_instr,
    )


if __name__ == "__main__":
    main()