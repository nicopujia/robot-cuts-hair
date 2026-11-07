"""Plan every arm move the web demo can play with DimOS, and save them as JSON.

Needs haircut_stack.py running. Usage (from this folder):
    ./py plan_library.py --out ../web/public/plans.json

The arm visits a fixed "ready" pose above the head between rows, so any row can
follow any other. For each cut height and row this plans, in order:
    ready -> above the first station (joint space)
    for each station: over to it above the hair and down into the hair, then up to cut height
    up out of the hair, then -> ready (joint space)
plus home -> ready and ready -> home. Sideways moves only happen above the hair, since
scissors dragged through hair would snag it. Everything is planned against the scanned
head and neck mesh and the shoulders, and every plan is really executed on the mock arm,
because DimOS always plans from the arm's current state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from clipper_path import Waypoint
from haircut import go_home, settled_pose, to_pose
from haircut_module import HaircutManipulationModule
from head_model import ROW_OFFSETS, SKULL_CENTER, WORLD_MESH, Station, scalp_stations, surface_tree, world_triangles
import numpy as np

from dimos.manipulation.manipulation_spec import ExecutionStatus, PlanResult
from dimos.manipulation.sdk import Arm
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.porcelain.dimos import Dimos

SHOULDERS = {"center": [0.47, 0.0, -0.27], "size": [0.42, 0.20, 0.14]}  # wide ear to ear
TRAVEL_SPEED = 0.2  # m/s, above the hair and on the way down into it
RAISE_SPEED = 0.04  # m/s, while the comb lifts the hair to the blades
RETRIES = 4
HAIR_MARGIN = 0.215  # metres from the skull centre, for moves between home and ready
SAMPLE_DT = 1 / 30


def place_body(manip: Any) -> None:
    for name in ("head", "shoulders"):
        manip.remove_obstacle(name)
    # The mesh is already in world coordinates.
    manip.add_obstacle("head", Pose(position=[0.0, 0.0, 0.0]), "mesh", mesh_path=str(WORLD_MESH))
    manip.add_obstacle("shoulders", Pose(position=SHOULDERS["center"]), "box", SHOULDERS["size"])


def with_hair_margin(manip: Any, plan: Any) -> Any:
    """Plan a free move with a sphere around the head big enough for the hair and scissors.

    The planner knows the head but not the hair or the scissors reaching out from the
    gripper, and a free move could otherwise swing them through the hair.
    """

    def planned() -> PlanResult:
        manip.add_obstacle("hair_margin", Pose(position=SKULL_CENTER.tolist()), "sphere", [HAIR_MARGIN])
        try:
            return plan()
        finally:
            manip.remove_obstacle("hair_margin")

    return planned


def sample_trajectory(plan: Any) -> dict[str, Any]:
    """Resample a planned trajectory at a fixed rate, keeping its timing."""
    traj = plan.trajectory
    names = list(traj.joint_names)
    arm_idx = [names.index(f"joint{i}") for i in range(1, 8)]
    times = np.array([p.time_from_start for p in traj.points], dtype=float)
    q = np.array([[p.positions[i] for i in arm_idx] for p in traj.points], dtype=float)
    t = np.arange(0.0, times[-1] + SAMPLE_DT, SAMPLE_DT)
    t[-1] = min(t[-1], times[-1])
    resampled = np.column_stack([np.interp(t, times, q[:, j]) for j in range(7)])
    return {"dt": SAMPLE_DT, "duration": float(times[-1]), "q": np.round(resampled, 5).tolist()}


def run_plan(arm: Arm, plan_fn: Any, label: str) -> dict[str, Any]:
    """Plan (retrying, since RRT is random), execute on the mock arm, return the samples."""
    last = ""
    for _ in range(RETRIES):
        planned: PlanResult = plan_fn()
        if planned.succeeded and planned.plan is not None:
            duration = planned.plan.trajectory.duration
            result = arm.rpc.execute(
                blocking=True, timeout=duration + 30, plan_id=planned.plan.plan_id
            )
            if result.status is not ExecutionStatus.COMPLETED:
                raise RuntimeError(f"{label}: execution {result.status.name}: {result.message}")
            samples = sample_trajectory(planned.plan)
            print(f"  {label}: {samples['duration']:.1f}s", flush=True)
            return samples
        last = str(planned.message)
        time.sleep(0.2)
    raise RuntimeError(f"{label}: {last}")


def plan_pose(arm: Arm, pose: PoseStamped, speed_scale: float = 0.6) -> Any:
    return lambda: arm.rpc.plan_to_poses({arm.info.id: pose}, speed_scale=speed_scale)


def plan_through(arm: Arm, manip: Any, points: list[Waypoint], speed: float) -> Any:
    """A Cartesian move from where the arm is now, through ``points``."""

    def plan() -> PlanResult:
        waypoints = [settled_pose(arm), *(to_pose(w) for w in points)]
        return manip.plan_cartesian_waypoints(
            arm.info.id, waypoints, max_linear_speed=speed, check_collision=True
        )

    return plan


def ready_pose() -> PoseStamped:
    # Above the head, tool straight down, wrist spun like the strokes.
    q = Quaternion.from_rotation_matrix(np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]))
    position = (SKULL_CENTER + [0.0, 0.0, 0.27]).tolist()
    return PoseStamped(frame_id="world", position=position, orientation=q)


def plan_joints(arm: Arm, q: list[float], speed_scale: float = 0.6) -> Any:
    target = JointState(name=list(arm.info.joint_names), position=q)
    return lambda: arm.rpc.plan_to_joints({arm.info.id: target}, speed_scale=speed_scale)


def go_to(arm: Arm, q: list[float]) -> None:
    """Joint-space move, skipped if already there (DimOS fails on zero-length plans)."""
    if np.max(np.abs(arm.joints() - np.array(q))) > 1e-3:
        run_plan(arm, plan_joints(arm, q), "reposition")
        time.sleep(0.2)


def plan_row(
    arm: Arm, manip: Any, row: list[Station], ready_q: list[float], seeds: list[list[float]], label: str
) -> dict[str, Any]:
    """Find an arm posture that works for every station, recording the whole row from it.

    A 7-joint arm can reach the same pose in many postures, and the Cartesian planner
    can't change posture on the way, so whether a row works depends on how the arm
    arrives. Each seed posture leads the IK to a different start.
    """
    last = ""
    for seed in seeds:
        try:
            go_to(arm, seed)
            run_plan(arm, plan_pose(arm, to_pose(row[0].air)), "try approach")
            time.sleep(0.2)
            start_q = arm.joints().tolist()
            stations = []
            for s, station in enumerate(row):
                points = [station.low] if s == 0 else [row[s - 1].air, station.air, station.low]
                travel = run_plan(arm, plan_through(arm, manip, points, TRAVEL_SPEED), f"{label} s{s + 1} travel")
                time.sleep(0.2)
                up = run_plan(arm, plan_through(arm, manip, [station.cut], RAISE_SPEED), f"{label} s{s + 1} raise")
                time.sleep(0.2)
                stations.append(
                    {"travel": travel, "raise": up, "depth": round(station.depth, 4), "rise": round(station.rise, 4)}
                )
        except RuntimeError as exc:
            last = str(exc)
            print(f"  posture failed: {last[:160]}", flush=True)
            continue
        lift = run_plan(arm, plan_through(arm, manip, [row[-1].air], TRAVEL_SPEED), f"{label} lift")
        time.sleep(0.2)
        leave = run_plan(arm, plan_joints(arm, ready_q), f"{label} leave")
        time.sleep(0.2)
        approach = run_plan(arm, plan_joints(arm, start_q), f"{label} approach")
        time.sleep(0.2)
        go_to(arm, ready_q)
        return {"approach": approach, "stations": stations, "lift": lift, "leave": leave, "start_q": start_q}
    raise RuntimeError(f"{label}: no working posture: {last}")


def plan_height(
    arm: Arm,
    manip: Any,
    tris: Any,
    tree: Any,
    clearance: float,
    yaw: float,
    ready_q: list[float],
    known: dict[int, list[float]],
) -> dict[str, Any]:
    """Plan all rows at one cut height; the arm starts and ends at the ready pose."""
    rows = scalp_stations(tris, tree, clearance, yaw_deg=yaw)
    home = [0.0] * 7
    out: dict[int, dict[str, Any]] = {}
    # Centre outward, so each row can start from its neighbour's working posture.
    for i in sorted(range(len(rows)), key=lambda k: abs(k - len(rows) // 2)):
        print(f" row {i + 1}: {len(rows[i])} stations", flush=True)
        neighbours = [out[j]["start_q"] for j in (i - 1, i + 1) if j in out]
        seeds = [*([known[i]] if i in known else []), *neighbours, ready_q, home]
        out[i] = plan_row(arm, manip, rows[i], ready_q, seeds, f"r{i + 1}")
        known[i] = out[i]["start_q"]
    for row in out.values():
        del row["start_q"]
    return {"clearance": clearance, "rows": [out[i] for i in range(len(rows))]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--yaw", type=float, default=90.0)
    parser.add_argument("--heights", type=float, nargs="+", default=[0.025, 0.03, 0.04])
    parser.add_argument("--drag-only", action="store_true", help="add only the deliberate drag to --out")
    parser.add_argument("--home-moves-only", action="store_true", help="re-plan only home<->ready in --out")
    args = parser.parse_args()

    app = Dimos.connect()
    try:
        arm = Arm.from_app(app)
        manip = app.get_module(HaircutManipulationModule.__name__)
        arm.close_gripper()
        for name in ("head", "shoulders"):
            manip.remove_obstacle(name)
        go_home(arm, 1.0)
        place_body(manip)
        tris = world_triangles()
        tree = surface_tree()
        print("obstacles:", list(manip.get_obstacles()), flush=True)
        if args.drag_only:
            # A deliberately bad move for the demo: the middle row, dragged through the hair.
            data = json.loads(args.out.read_text())
            ready_q = data["home_to_ready"]["q"][-1]
            height = 0
            rows = scalp_stations(tris, surface_tree(), data["heights"][height]["clearance"], yaw_deg=args.yaw)
            row = rows[len(rows) // 2]
            go_to(arm, ready_q)
            run_plan(arm, plan_pose(arm, to_pose(row[0].air)), "try approach")
            time.sleep(0.2)
            start_q = arm.joints().tolist()
            down = run_plan(arm, plan_through(arm, manip, [row[0].cut], TRAVEL_SPEED), "down")
            time.sleep(0.2)
            stroke = run_plan(arm, plan_through(arm, manip, [s.cut for s in row[1:]], 0.08), "drag")
            time.sleep(0.2)
            lift = run_plan(arm, plan_through(arm, manip, [row[-1].air], TRAVEL_SPEED), "lift")
            time.sleep(0.2)
            leave = run_plan(arm, plan_joints(arm, ready_q), "leave")
            time.sleep(0.2)
            approach = run_plan(arm, plan_joints(arm, start_q), "approach")
            data["drag"] = {"height": height, "row": len(rows) // 2, "approach": approach, "down": down,
                            "stroke": stroke, "lift": lift, "leave": leave}
            time.sleep(0.2)
            go_to(arm, ready_q)
            args.out.write_text(json.dumps(data, separators=(",", ":")))
            print(f"updated {args.out}")
            return 0
        if args.home_moves_only:
            data = json.loads(args.out.read_text())
            ready_q = data["home_to_ready"]["q"][-1]
            data["home_to_ready"] = run_plan(arm, with_hair_margin(manip, plan_joints(arm, ready_q)), "home->ready")
            time.sleep(0.5)
            data["ready_to_home"] = run_plan(arm, with_hair_margin(manip, plan_joints(arm, np.zeros(7).tolist())), "ready->home")
            args.out.write_text(json.dumps(data, separators=(",", ":")))
            print(f"updated {args.out}")
            return 0
        go_ready = run_plan(arm, with_hair_margin(manip, plan_pose(arm, ready_pose())), "home->ready")
        time.sleep(0.5)
        ready_q = go_ready["q"][-1]
        heights = []
        known: dict[int, list[float]] = {}
        for clearance in args.heights:
            print(f"cut height {clearance * 100:.1f} cm", flush=True)
            heights.append(plan_height(arm, manip, tris, tree, clearance, args.yaw, ready_q, known))
        go_home_plan = run_plan(arm, with_hair_margin(manip, plan_joints(arm, np.zeros(7).tolist())), "ready->home")
        data = {
            "source": "DimOS RoboPlan planner, mock xArm7",
            "joints": [f"joint{i}" for i in range(1, 8)],
            "shoulders": SHOULDERS,
            "row_offsets": ROW_OFFSETS,
            "yaw_deg": args.yaw,
            "travel_speed": TRAVEL_SPEED,
            "raise_speed": RAISE_SPEED,
            "home_to_ready": go_ready,
            "ready_to_home": go_home_plan,
            "heights": heights,
        }
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(data, separators=(",", ":")))
            print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
        return 0
    finally:
        app.stop()


if __name__ == "__main__":
    sys.exit(main())
