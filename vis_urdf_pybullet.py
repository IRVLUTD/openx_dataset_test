#!/usr/bin/env python3
"""
vis_urdf_pybullet_multi_robot.py

A small, flexible PyBullet URDF visualizer that supports different robots
(e.g., Jaco and Franka Panda) by:
- Selecting a robot preset (--robot jaco|panda) OR custom joint name lists.
- Mapping an input joint vector onto URDF joints by NAME (robust).
- Optionally setting gripper/finger joints.
- Printing joint states for debugging.

Usage examples
--------------
# Jaco (6 arm + 2 gripper values)
python vis_urdf_pybullet_multi_robot.py --robot jaco \
  --urdf robots/jaco_description/urdf/jaco_arm.urdf \
  --q -1.8585522 4.022645 1.7151797 -0.3398039 1.0661805 -0.8476771 0.41148704 0.40163106

# Franka Panda (7 arm + 2 gripper values)
python vis_urdf_pybullet_multi_robot.py --robot panda \
  --urdf robots/panda_description/urdf/panda.urdf \
  --q 0.0 -0.5 0.0 -2.0 0.0 2.0 0.8 0.02 0.02

# Custom: specify joint names explicitly
python vis_urdf_pybullet_multi_robot.py --urdf path/to/robot.urdf \
  --arm_joint_names joint1 joint2 joint3 joint4 \
  --gripper_joint_names grip1 grip2 \
  --q 0.1 0.2 0.3 0.4 0.01 0.01
"""

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p


# -----------------------------
# Utilities
# -----------------------------

def wrap_to_pi(x: float) -> float:
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def dump_joints(robot: int) -> None:
    rows = []
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        name = info[1].decode("utf-8")
        jtype = info[2]
        q = p.getJointState(robot, j)[0]
        rows.append((j, name, jtype, q))
    for r in rows:
        print(f"{r[0]:2d}  {r[1]:40s}  type={r[2]}  q={r[3]: .4f}")


def build_name_to_idx(robot: int) -> Dict[str, int]:
    """Map movable joint name -> joint index."""
    name_to_idx = {}
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        jname = info[1].decode("utf-8")
        jtype = info[2]
        if jtype != p.JOINT_FIXED:
            name_to_idx[jname] = j
    return name_to_idx


def apply_joint_vector_by_name(
    robot: int,
    name_to_idx: Dict[str, int],
    arm_joint_names: List[str],
    gripper_joint_names: List[str],
    q: List[float],
    wrap_angles: bool = False,
) -> None:
    """
    Applies q to the robot by name:
      q[:len(arm_joint_names)] -> arm joints
      q[len(arm_joint_names):len(arm)+len(gripper)] -> gripper joints (if provided)
    """
    q = list(q)
    n_arm = len(arm_joint_names)
    n_grip = len(gripper_joint_names)

    if len(q) < n_arm:
        raise ValueError(f"Provided --q has length {len(q)}, but needs at least {n_arm} for the arm joints.")

    # Arm
    for i, jn in enumerate(arm_joint_names):
        if jn not in name_to_idx:
            print(f"WARNING: arm joint not found in URDF: {jn}")
            continue
        val = float(q[i])
        # Optional wrapping (helps if dataset uses [0,2pi) / unwrapped angles)
        if wrap_angles:
            val = wrap_to_pi(val)
        p.resetJointState(robot, name_to_idx[jn], val)

    # Gripper
    if n_grip > 0:
        start = n_arm
        end = n_arm + n_grip
        if len(q) < end:
            print(f"NOTE: --q has only {len(q)} values; expected {end} to also set gripper joints. "
                  f"Skipping gripper.")
            return
        for i, jn in enumerate(gripper_joint_names):
            if jn not in name_to_idx:
                print(f"WARNING: gripper joint not found in URDF: {jn}")
                continue
            p.resetJointState(robot, name_to_idx[jn], float(q[start + i]))


def suggest_joints(name_to_idx: Dict[str, int]) -> Tuple[List[str], List[str]]:
    """
    Heuristic fallback if you don't provide --robot preset or explicit joint lists.
    - Arm joints: joints whose name contains common arm patterns and are not fingers.
    - Gripper joints: joints containing finger/gripper.
    """
    names = sorted(name_to_idx.keys())
    grip = [n for n in names if any(s in n.lower() for s in ["finger", "gripper"])]
    arm = [n for n in names if n not in grip]
    return arm, grip


# -----------------------------
# Robot presets
# -----------------------------

PRESETS = {
    # Your Jaco URDF example
    "jaco": {
        "arm": ["jaco_joint_1", "jaco_joint_2", "jaco_joint_3", "jaco_joint_4", "jaco_joint_5", "jaco_joint_6"],
        # NOTE: these names vary by URDF. We'll try common ones; missing ones will warn.
        "gripper": ["jaco_finger_joint_1", "jaco_finger_joint_2", "jaco_finger_joint_3"],
    },
    # Franka Panda (typical URDF joint names)
    "panda": {
        "arm": ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"],
        "gripper": ["panda_finger_joint1", "panda_finger_joint2"],
    },
}


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, help="Path to URDF file")
    ap.add_argument("--robot", default=None, choices=[None, "jaco", "panda"],
                    help="Robot preset (optional). If omitted, use explicit joint names or heuristic.")
    ap.add_argument("--q", type=float, nargs="+", required=True,
                    help="Joint values: arm first, then gripper (if provided).")
    ap.add_argument("--arm_joint_names", nargs="*", default=None,
                    help="Explicit arm joint names (overrides preset if provided).")
    ap.add_argument("--gripper_joint_names", nargs="*", default=None,
                    help="Explicit gripper/finger joint names (overrides preset if provided).")
    ap.add_argument("--wrap", action="store_true",
                    help="Wrap arm joint angles to (-pi, pi] before setting (useful for some datasets).")
    ap.add_argument("--print_joints_only", action="store_true",
                    help="Print movable joint names and exit.")
    args = ap.parse_args()

    p.connect(p.GUI)
    robot = p.loadURDF(args.urdf, useFixedBase=True)

    name_to_idx = build_name_to_idx(robot)

    print("Movable joints in URDF:")
    for k in sorted(name_to_idx.keys()):
        print(" ", k)

    if args.print_joints_only:
        return

    # Choose joint name lists
    if args.arm_joint_names is not None and len(args.arm_joint_names) > 0:
        arm_joint_names = args.arm_joint_names
        gripper_joint_names = args.gripper_joint_names or []
    elif args.robot is not None:
        arm_joint_names = PRESETS[args.robot]["arm"]
        gripper_joint_names = PRESETS[args.robot]["gripper"]
    else:
        # Heuristic fallback
        arm_joint_names, gripper_joint_names = suggest_joints(name_to_idx)
        print("\nHeuristic joint selection (override with --arm_joint_names/--gripper_joint_names if wrong):")
        print("  Arm joints (ordered):", arm_joint_names)
        print("  Gripper joints (ordered):", gripper_joint_names)

    print("\nUsing arm joints:", arm_joint_names)
    print("Using gripper joints:", gripper_joint_names)

    apply_joint_vector_by_name(
        robot,
        name_to_idx,
        arm_joint_names=arm_joint_names,
        gripper_joint_names=gripper_joint_names,
        q=args.q,
        wrap_angles=args.wrap,
    )

    print("\nJoint states after setting:")
    dump_joints(robot)

    # Keep GUI open
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
