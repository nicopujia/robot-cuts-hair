"""Independent check of plans.json: forward kinematics on every sample, not using DimOS.

Reports tool clearance to the real head mesh, whether moves join up (any row can follow
any other, so all rows must share one ready posture), and whether the tool ever moves
sideways while it's low enough to be in the hair, which would snag it in real life.
Exits non-zero on any of those.

    uv run --no-project --python 3.12 --with numpy --with scipy --with trimesh \\
        python verify_plans.py ../web/public assets/head_world.stl
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import trimesh

JUMP_LIMIT = 0.02  # rad; at 30 Hz anything bigger between moves is a visible jump
MIN_TOOL_CLEARANCE = 0.003  # metres; a margin well above the error of the surface sampling

# Tool outline in the tool frame, metres. Keep in sync with buildTool in web/src/main.js.
PLANE_Z = -0.0167  # blades sit at the finger pads
PIVOT_X = 0.05
BLADE_LEN = 0.07
OPEN_ANGLE = 18.0
BLADE_CENTER = np.array([0.085, 0.0, PLANE_Z])
COMB_X0, COMB_SPINE_X = 0.066, 0.126  # a flat fan inside the open jaws, from near the pivot out to its spine
COMB_FAN_FROM, COMB_FAN, COMB_FAN_MARGIN = 0.0476, 0.317, 0.003
COMB_TOP = PLANE_Z + 0.0015  # the lower blade rides on it
FINGER_POINTS = np.array(  # fingertips, in the tool frame
    [[0.0, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, -0.03, 0.0], [0.012, 0.0, 0.0], [-0.012, 0.0, 0.0]]
)
ABOVE_HAIR = 0.04  # the tool may only move sideways with everything this far off the scalp
SIDEWAYS_LIMIT = 0.012  # m/s; slower than this counts as not moving sideways


def tool_points() -> np.ndarray:
    points = []
    pivot = np.array([PIVOT_X, 0.0, PLANE_Z])
    for angle in np.radians([-OPEN_ANGLE, 0.0, OPEN_ANGLE]):
        c, s = np.cos(angle), np.sin(angle)
        for r in np.linspace(0.0, BLADE_LEN, 11):  # blades
            points.append(pivot + [r * c, r * s, 0.0])
            points.append(pivot + [r * c, r * s, 0.003])
        for r in np.linspace(0.0, PIVOT_X, 7):  # handles, back to the fingers
            points.append(pivot + [-r * c, -r * s, 0.0])
    for x in np.linspace(COMB_X0, COMB_SPINE_X, 13):  # the comb
        half = (x - COMB_FAN_FROM) * COMB_FAN - COMB_FAN_MARGIN
        for y in (-half, 0.0, half):
            points.append([x, y, COMB_TOP + 0.0025])
    for x in np.linspace(0.0, PIVOT_X + BLADE_LEN + 0.005, 13):  # guide rail
        points.append([x, 0.0, PLANE_Z - 0.008])
    return np.array(points)


def transform(xyz, rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    m = np.eye(4)
    m[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    m[:3, 3] = xyz
    return m


def rotation(axis, angle) -> np.ndarray:
    x, y, z = np.array(axis) / np.linalg.norm(axis)
    c, s, v = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    m = np.eye(4)
    m[:3, :3] = [
        [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
        [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
        [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
    ]
    return m


def tcp_chain(kinematics: dict) -> list[dict]:
    by_child = {j["child"]: j for j in kinematics["joints"]}
    chain, link = [], "link_tcp"
    while link in by_child:
        chain.append(by_child[link])
        link = by_child[link]["parent"]
    return chain[::-1]


def fk(chain: list[dict], q) -> np.ndarray:
    m = np.eye(4)
    for j in chain:
        m = m @ transform(j["xyz"], j["rpy"])
        if j["type"] != "fixed":
            m = m @ rotation(j["axis"], q[int(j["name"].removeprefix("joint")) - 1])
    return m


def main() -> int:
    public = Path(sys.argv[1])
    plans = json.loads((public / "plans.json").read_text())
    chain = tcp_chain(json.loads((public / "robot/kinematics.json").read_text()))
    # Dense samples of the scan's surface stand in for exact point-to-triangle distance.
    surface, _ = trimesh.sample.sample_surface(trimesh.load(sys.argv[2], force="mesh"), 400_000, seed=1)
    tree = cKDTree(surface)
    tool = tool_points()
    failures = []

    def jump(a, b, label):
        gap = float(np.max(np.abs(np.array(a) - np.array(b))))
        if gap > JUMP_LIMIT:
            failures.append(f"jump of {gap:.3f} rad at {label}")
        return gap

    ready = plans["home_to_ready"]["q"][-1]
    jump(plans["home_to_ready"]["q"][0], [0] * 7, "home_to_ready start")
    jump(plans["ready_to_home"]["q"][0], ready, "ready_to_home start")
    jump(plans["ready_to_home"]["q"][-1], [0] * 7, "ready_to_home end")

    def sideways_in_hair(move: dict, label: str) -> None:
        """Free moves too: nothing may swing through the hair on the way anywhere."""
        worst, last = 0.0, None
        for q in move["q"]:
            m = fk(chain, q)
            gap = float(tree.query((m[:3, :3] @ tool.T).T + m[:3, 3])[0].min())
            blade = m[:3, :3] @ BLADE_CENTER + m[:3, 3]
            if last is not None and gap < ABOVE_HAIR:
                worst = max(worst, float(np.linalg.norm(blade - last)) / move["dt"])
            last = blade
        print(f"{label}: fastest move with the tool in the hair {worst * 1000:.1f} mm/s")
        if worst > SIDEWAYS_LIMIT:
            failures.append(f"{label} moves through the hair at {worst * 1000:.0f} mm/s")

    # The deliberate drag is allowed through the hair, but not into the head.
    if "drag" in plans:
        gap = np.inf
        for name in ("approach", "down", "stroke", "lift", "leave"):
            for q in plans["drag"][name]["q"]:
                m = fk(chain, q)
                gap = min(gap, float(tree.query((m[:3, :3] @ tool.T).T + m[:3, 3])[0].min()))
        print(f"deliberate drag: scissors and comb closest {gap * 100:.2f} cm")
        if gap < MIN_TOOL_CLEARANCE:
            failures.append("the deliberate drag gets too close to the head")

    sideways_in_hair(plans["home_to_ready"], "home to ready")
    sideways_in_hair(plans["ready_to_home"], "ready to home")

    for h in plans["heights"]:
        target = h["clearance"]
        at_cut, scissor_min, finger_min, in_hair_sideways = [], np.inf, np.inf, 0.0
        for i, row in enumerate(h["rows"]):
            label = f"{target * 100:.1f} cm row {i + 1}"
            moves = [("approach", row["approach"])]
            for s, station in enumerate(row["stations"]):
                moves += [(f"station {s + 1} travel", station["travel"]), (f"station {s + 1} raise", station["raise"])]
            moves += [("lift", row["lift"]), ("leave", row["leave"])]
            jump(row["approach"]["q"][0], ready, f"{label} start")
            jump(row["leave"]["q"][-1], ready, f"{label} end")
            for (_, a), (name, b) in zip(moves, moves[1:], strict=False):
                jump(b["q"][0], a["q"][-1], f"{label} into {name}")
            for name, move in moves:
                last = None
                for q in move["q"]:
                    m = fk(chain, q)
                    world = (m[:3, :3] @ tool.T).T + m[:3, 3]
                    gap = float(tree.query(world)[0].min())
                    fingers = float(tree.query((m[:3, :3] @ FINGER_POINTS.T).T + m[:3, 3])[0].min())
                    scissor_min = min(scissor_min, gap)
                    finger_min = min(finger_min, fingers)
                    blade = m[:3, :3] @ BLADE_CENTER + m[:3, 3]
                    if last is not None:
                        step = (blade - last) / move["dt"]
                        axis = m[:3, 2]
                        sideways = float(np.linalg.norm(step - step.dot(axis) * axis))
                        if min(gap, fingers) < ABOVE_HAIR:
                            in_hair_sideways = max(in_hair_sideways, sideways)
                    last = blade
                if name.endswith("raise"):
                    at_cut.append(float(tree.query(blade)[0]))
        print(
            f"cut height {target * 100:.1f} cm: blades close {min(at_cut) * 100:.2f}-{max(at_cut) * 100:.2f} cm off "
            f"scalp; scissors and comb closest {scissor_min * 100:.2f} cm; fingertips closest {finger_min * 100:.2f} cm; "
            f"fastest sideways with the tool in the hair {in_hair_sideways * 1000:.1f} mm/s"
        )
        if scissor_min < MIN_TOOL_CLEARANCE:
            failures.append(f"scissors or comb within {MIN_TOOL_CLEARANCE * 1000:.0f} mm of the head at cut height {target * 100:.1f} cm")
        if finger_min < MIN_TOOL_CLEARANCE:
            failures.append(f"fingers within {MIN_TOOL_CLEARANCE * 1000:.0f} mm of the head at cut height {target * 100:.1f} cm")
        if in_hair_sideways > SIDEWAYS_LIMIT:
            failures.append(f"tool moved sideways in the hair at {in_hair_sideways * 1000:.0f} mm/s at cut height {target * 100:.1f} cm")

    print("FAIL:" if failures else "All checks passed.")
    for f in failures[:20]:
        print(" ", f)
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
