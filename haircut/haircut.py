"""Put a head in the scene and make the mock xArm7 trace a clipper path over it.

Needs haircut_stack.py running. Usage (from this folder):
    ./py haircut.py                          # place head, run every clipper row
    ./py haircut.py --center 0.45 0 0.1 --stroke y
    ./py haircut.py --scan                   # try a grid of head placements quickly

Each row: a collision-checked joint-space move to just above the row's start,
then one collision-checked Cartesian path (lower, stroke, lift) that is executed.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import threading
import time
from typing import Any

from clipper_path import Waypoint, clipper_rows, min_clearance
from haircut_module import HaircutManipulationModule
import numpy as np

from dimos.manipulation.manipulation_spec import ExecutionStatus
from dimos.manipulation.sdk import Arm, MotionError
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.porcelain.dimos import Dimos

HEAD_RADIUS = 0.10  # 20 cm across
CLEARANCE = 0.02
SKIN_RGBA = [0.87, 0.68, 0.55, 1.0]


def to_pose(w: Waypoint) -> PoseStamped:
    q = Quaternion.from_rotation_matrix(w.rotation)
    return PoseStamped(frame_id="world", position=w.position.tolist(), orientation=q)


def place_head(manip: Any, center: np.ndarray) -> None:
    pose = Pose(position=center.tolist())
    if "head" not in manip.get_obstacles():
        manip.add_obstacle("head", pose, "sphere", [HEAD_RADIUS])
    manip.update_obstacle("head", pose, "sphere", [HEAD_RADIUS], color=SKIN_RGBA)


def go_home(arm: Arm, speed_scale: float) -> None:
    if np.max(np.abs(arm.joints())) > 1e-3:  # planning a zero-length move fails in DimOS
        arm.home(speed_scale=speed_scale)


def settled_pose(arm: Arm, timeout: float = 2.0) -> PoseStamped:
    """The TCP pose once telemetry stops changing; right after a move it can lag."""
    last = arm.pose()
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.15)
        pose = arm.pose()
        moved = np.linalg.norm(np.subtract(pose.position.to_tuple(), last.position.to_tuple()))
        if moved < 1e-5:
            return pose
        last = pose
    return last


PLANNER_OVERRIDES: dict[str, float] = {}


class ClearanceProbe:
    """Samples the live TCP pose during execution and records its distance to the scalp."""

    def __init__(self, arm: Arm, center: np.ndarray) -> None:
        self.arm, self.center, self.samples = arm, center, []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            p = np.array(self.arm.pose().position.to_tuple())
            self.samples.append(float(np.linalg.norm(p - self.center)) - HEAD_RADIUS)
            time.sleep(0.05)

    def __enter__(self) -> ClearanceProbe:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def run_row(
    arm: Arm, manip: Any, row: list[Waypoint], speed: float, center: np.ndarray
) -> tuple[str, list[float]]:
    """Approach, plan and execute one row. Returns 'ok' or why it failed, plus TCP clearances."""
    start = to_pose(row[0])
    try:
        arm.move_pose(
            start.position.to_tuple(), orientation=start.orientation.to_tuple(), speed_scale=0.6
        )
    except MotionError as exc:
        return f"approach failed: {getattr(exc.result, 'message', exc)}", []
    waypoints = [settled_pose(arm), *(to_pose(w) for w in row[1:])]
    planned = manip.plan_cartesian_waypoints(
        arm.info.id, waypoints, max_linear_speed=speed, check_collision=True,
        planner_overrides=PLANNER_OVERRIDES,
    )
    if not planned.succeeded or planned.plan is None:
        return f"Cartesian planning failed: {planned.message}", []
    duration = planned.plan.trajectory.duration
    with ClearanceProbe(arm, center) as probe:
        result = arm.rpc.execute(
            blocking=True, timeout=duration + 30, plan_id=planned.plan.plan_id
        )
    if result.status is not ExecutionStatus.COMPLETED:
        return f"execution {result.status.name}: {result.message}", probe.samples
    return "ok", probe.samples


def haircut(
    arm: Arm, manip: Any, center: np.ndarray, stroke: str, sweep: tuple[float, float],
    n_rows: int, speed: float, yaw: float = 0.0, tilt: float = 1.0, verbose: bool = True,
) -> list[str]:
    manip.remove_obstacle("head")
    manip.set_camera((center + [0.08, -0.72, 0.42]).tolist(), (center + [-0.14, 0.0, 0.14]).tolist())
    go_home(arm, 1.0)
    place_head(manip, center)
    rows = clipper_rows(center, HEAD_RADIUS, CLEARANCE, stroke, sweep, n_rows=n_rows, yaw_deg=yaw, tilt_scale=tilt)
    if verbose:
        gap = min(min_clearance(r, center, HEAD_RADIUS) for r in rows)
        print(f"{len(rows)} rows, {sum(len(r) for r in rows)} waypoints, "
              f"min clearance {gap * 100:.2f} cm; obstacles: {list(manip.get_obstacles())}")
    results = []
    for i, row in enumerate(rows, 1):
        t0 = time.time()
        status, clearances = run_row(arm, manip, row, speed, center)
        results.append(status)
        if verbose:
            gap = f", live TCP clearance min {min(clearances) * 100:.2f} cm" if clearances else ""
            print(f"row {i}: {status[:200]} ({time.time() - t0:.1f}s{gap})", flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--center", type=float, nargs=3, default=[0.40, 0.0, 0.05])
    parser.add_argument("--stroke", choices=["x", "y"], default="y")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--sweep", type=float, nargs=2, default=[-55.0, 55.0])
    parser.add_argument("--speed", type=float, default=0.08, help="max TCP speed, m/s")
    parser.add_argument("--yaw", type=float, default=180.0, help="clipper spin about its axis, deg")
    parser.add_argument("--tilt", type=float, default=0.6, help="1 = tool normal to scalp")
    parser.add_argument("--scan", action="store_true")
    args = parser.parse_args()
    sweep = (args.sweep[0], args.sweep[1])

    app = Dimos.connect()
    try:
        arm = Arm.from_app(app)
        manip = app.get_module(HaircutManipulationModule.__name__)
        arm.close_gripper()
        if args.scan:
            for stroke, tilt, yaw, x, z in itertools.product(
                "yx", (0.6, 0.4), (180,), (0.40, 0.46), (0.05,)
            ):
                res = haircut(arm, manip, np.array([x, 0.0, z]), stroke, sweep, args.rows, 0.5,
                              yaw=yaw, tilt=tilt, verbose=False)
                fails = [f"row{i}: {r[-90:]}" for i, r in enumerate(res, 1) if r != "ok"]
                print(f"stroke={stroke} tilt={tilt} yaw={yaw} center=({x:.2f}, 0, {z:+.2f}): "
                      f"{res.count('ok')}/{len(res)} ok {fails}", flush=True)
            return 0

        results = haircut(arm, manip, np.array(args.center), args.stroke, sweep, args.rows,
                          args.speed, yaw=args.yaw, tilt=args.tilt)
        try:
            arm.move_linear(dz=0.05, check_collision=True)
        except MotionError:
            pass
        manip.remove_obstacle("head")
        go_home(arm, 0.5)
        place_head(manip, np.array(args.center))
        return 0 if all(r == "ok" for r in results) else 1
    finally:
        app.stop()


if __name__ == "__main__":
    sys.exit(main())
