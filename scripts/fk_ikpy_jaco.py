#!/usr/bin/env python3
"""
fk_ikpy_jaco_complete.py

Compute forward kinematics (FK) for Kinova Jaco (j2n6s300) using ikpy, and compare
against an example OpenX jaco_play ee_pos = [x,y,z,qx,qy,qz,qw].

This version fixes the issues you hit:
- URDF root link name is not necessarily "base_link": we auto-detect a root link and pass base_elements.
- ikpy does not support URDF joint type "continuous": use a patched URDF where continuous->revolute.
- ikpy forward_kinematics() (in many versions) does NOT accept end_effector_index: it returns FK to the last link.
- ikpy active_links_mask can be misleading: we map joints by NAME explicitly to avoid ordering/activation issues.

Usage:
  python fk_ikpy_jaco_complete.py --urdf "C:\\path\\to\\jaco_ikpy.urdf"

Optional:
  python fk_ikpy_jaco_complete.py --urdf "...\\jaco_ikpy.urdf" --base_link "j2n6s300_link_base"
  python fk_ikpy_jaco_complete.py --urdf "...\\jaco_ikpy.urdf" --print_links
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from ikpy.chain import Chain


# ---- EDIT THESE IF YOUR URDF USES DIFFERENT JOINT NAMES ----
# These are the 6 arm joints in dataset order (very likely).
JOINT_NAME_ORDER = [
    "jaco_joint_1",
    "jaco_joint_2",
    "jaco_joint_3",
    "jaco_joint_4",
    "jaco_joint_5",
    "jaco_joint_6",
]


def read_urdf_links(urdf_path: str):
    """Return list of URDF <link name=...>."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    return [e.attrib["name"] for e in root.findall("link")]


def read_urdf_parent_child_links(urdf_path: str):
    """Return (links, parent_links, child_links) from URDF joints."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = [e.attrib["name"] for e in root.findall("link")]
    parent_links, child_links = [], []

    for j in root.findall("joint"):
        p = j.find("parent")
        c = j.find("child")
        if p is not None:
            parent_links.append(p.attrib["link"])
        if c is not None:
            child_links.append(c.attrib["link"])

    return links, parent_links, child_links


def guess_root_link(links, child_links):
    """Root link is a link that never appears as a joint child."""
    child_set = set(child_links)
    candidates = [l for l in links if l not in child_set]
    if candidates:
        return candidates[0], candidates
    return links[0], links


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (x,y,z,w)."""
    m = R
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= (np.linalg.norm(q) + 1e-12)
    return q


def quat_angle_error_deg(q1_xyzw, q2_xyzw) -> float:
    """Angular difference between two quats in degrees."""
    q1 = np.asarray(q1_xyzw, dtype=np.float64)
    q2 = np.asarray(q2_xyzw, dtype=np.float64)
    q1 /= (np.linalg.norm(q1) + 1e-12)
    q2 /= (np.linalg.norm(q2) + 1e-12)
    dot = float(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))
    ang = 2.0 * np.arccos(dot)
    return float(np.degrees(ang))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, help="Path to patched URDF (continuous->revolute) for ikpy")
    ap.add_argument("--base_link", default=None, help="Override root/base link name")
    ap.add_argument("--print_links", action="store_true", help="Print chain link names and exit")
    args = ap.parse_args()

    urdf_path = str(Path(args.urdf))
    links, parent_links, child_links = read_urdf_parent_child_links(urdf_path)

    if args.base_link is None:
        base_link, candidates = guess_root_link(links, child_links)
        print("Auto-detected base/root link:", base_link)
        if len(candidates) > 1:
            print("Other possible roots (first 10):", candidates[:10])
    else:
        if args.base_link not in links:
            raise ValueError(f"--base_link '{args.base_link}' not found. Example links: {links[:20]}")
        base_link = args.base_link
        print("Using user-specified base/root link:", base_link)

    # Build chain from URDF, starting at root link
    chain = Chain.from_urdf_file(urdf_path, base_elements=[base_link])

    # Print chain links
    print("\nChain links (index: name):")
    for i, link in enumerate(chain.links):
        print(f"  {i:2d}: {getattr(link, 'name', f'link_{i}')}")
    print("Total links:", len(chain.links))
    print("Active links mask:", chain.active_links_mask)

    if args.print_links:
        return

    # Build mapping from ikpy link name -> index
    link_name_to_idx = {getattr(link, "name", ""): i for i, link in enumerate(chain.links)}

    # Verify all joint names exist in chain
    missing = [n for n in JOINT_NAME_ORDER if n not in link_name_to_idx]
    if missing:
        raise RuntimeError(
            "These JOINT_NAME_ORDER entries were not found in the ikpy chain:\n"
            f"{missing}\n\n"
            "Fix: edit JOINT_NAME_ORDER at top of this script to match your chain names.\n"
        )

    # ---- Example OpenX jaco_play sample ----
    joint_pos8 = np.array([
        -1.8585522,  4.022645,   1.7151797, -0.3398039,
         1.0661805, -0.8476771,  0.41148704, 0.40163106
    ], dtype=np.float64)

    # Dataset ee_pos: [x, y, z, qx, qy, qz, qw]
    ee_obs = np.array([
         9.6705541e-02, -5.1346838e-01,  2.7464855e-01,
         2.0960305e-04, -1.3287666e-02,  9.9257147e-01,  1.2093488e-01
    ], dtype=np.float64)

    # Construct ikpy joint vector (length == number of links)
    # Set only the 6 arm joints explicitly; leave everything else as zero/neutral.
    full = np.zeros(len(chain.links), dtype=np.float64)
    for j, joint_name in enumerate(JOINT_NAME_ORDER):
        idx = link_name_to_idx[joint_name]
        full[idx] = float(joint_pos8[j])

    # Compute FK to the LAST link (ikpy default). No end_effector_index in many versions.
    T = chain.forward_kinematics(full)

    pos_fk = np.asarray(T[:3, 3], dtype=np.float64)
    quat_fk = rot_to_quat_xyzw(np.asarray(T[:3, :3], dtype=np.float64))

    print("\nFK result (base -> end link):")
    print("pos_fk (m):", pos_fk)
    print("quat_fk (xyzw):", quat_fk)
    print("\nT:\n", np.array(T))

    # Compare with dataset EE (note: may be in different frame than URDF base!)
    pos_obs = ee_obs[:3]
    quat_obs = ee_obs[3:]  # assumed xyzw

    pos_err = pos_fk - pos_obs
    pos_l2 = np.linalg.norm(pos_err)
    ang_err_deg = quat_angle_error_deg(quat_obs, quat_fk)

    print("\nCompare to dataset ee_pos (assuming same frame and xyzw quat):")
    print("pos_obs:", pos_obs)
    print("pos_err (m):", pos_err, "L2:", pos_l2)
    print("quat_obs (xyzw):", quat_obs / (np.linalg.norm(quat_obs) + 1e-12))
    print("angle_err (deg):", ang_err_deg)

    print("\nNote:")
    print("- If errors are large but roughly constant over time, your dataset EE is likely in a world frame,")
    print("  while FK is in the URDF base frame. Then you should fit a fixed base->world transform and re-check.")


if __name__ == "__main__":
    main()
