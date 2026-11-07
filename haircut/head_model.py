"""Realistic head: life-size placement of the scan, a scalp map, and tool paths over the scalp.

The scan (Lee Perry-Smith, CC-BY 3.0) uses arbitrary units with y up and the face toward +z.
Placed in the DimOS world, the face looks toward -y, the crown is up, and the middle of the
skull sits at SKULL_CENTER.

Build the derived files (from this folder):
    ./py head_model.py
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from clipper_path import Waypoint, _frame
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
import trimesh
from verify_plans import FINGER_POINTS, tool_points

HERE = Path(__file__).parent
SCAN = HERE.parent / "web/public/head/LeePerrySmith.glb"
PUBLIC_HEAD = HERE.parent / "web/public/head"
ASSETS = HERE / "assets"
WORLD_MESH = ASSETS / "head_world.stl"

SKULL_CENTER = np.array([0.47, 0.0, 0.05])
SCALE = 0.0425  # metres per scan unit: chin to crown is about 23 cm
# Scan axes to world axes: x stays, scan z (face) -> world -y, scan y (up) -> world z.
AXES = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
MAP_LON, MAP_LAT = 180, 90  # scalp map resolution
ROW_OFFSETS = (-0.05, -0.025, 0.0, 0.025, 0.05)  # metres, ear to ear
# Middle of the blades in the tool frame. The fingers hold the scissor rings at the pads,
# 16.7 mm behind link_tcp, and the blades reach out 5-12 cm along tool x from there.
BLADE_CENTER = np.array([0.085, 0.0, -0.0167])
TOOL_POINTS = tool_points()
MAX_DEPTH = 0.022  # metres; how far below the cut height the comb goes to get under the hair
LOW_COMB_GAP = 0.005  # nearest the comb and blades may get to the scalp at the bottom
LOW_FINGER_GAP = 0.006  # the same for the gripper's fingers
ABOVE_HAIR = 0.075  # how far off the scalp the tool rises before moving sideways: enough for hair to slide off the comb
BACK_MARGIN = 0.005  # metres of extra height over the back of the head


def load_scan() -> trimesh.Trimesh:
    return trimesh.load(SCAN, force="mesh", process=False)


def fit_skull(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Least-squares sphere through the cranium (scan units); returns its centre."""
    top = vertices[vertices[:, 1] > 2.0]
    a = np.column_stack([2 * top, np.ones(len(top))])
    b = (top**2).sum(axis=1)
    solution = np.linalg.lstsq(a, b, rcond=None)[0]
    return solution[:3]


def placement(mesh: trimesh.Trimesh) -> dict[str, list[float] | float]:
    translation = SKULL_CENTER - AXES @ (fit_skull(mesh.vertices) * SCALE)
    return {"scale": SCALE, "axes": AXES.tolist(), "translation": translation.tolist()}


def to_world(vertices: NDArray[np.float64], place: dict) -> NDArray[np.float64]:
    return (np.array(place["axes"]) @ (vertices * place["scale"]).T).T + np.array(place["translation"])


def ray_hits(
    tris: NDArray[np.float64], origin: NDArray[np.float64], directions: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Farthest hit distance along each ray from one origin (Moller-Trumbore). NaN if none."""
    v0 = tris[:, 0]
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0
    t0 = origin - v0
    out = np.full(len(directions), np.nan)
    for start in range(0, len(directions), 256):
        d = directions[start : start + 256][:, None, :]
        h = np.cross(d, e2[None])
        det = (e1[None] * h).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / det
            u = (t0[None] * h).sum(-1) * inv
            q = np.cross(t0[None], e1[None])
            v = (d * q).sum(-1) * inv
            t = (e2[None] * q).sum(-1) * inv
        hit = (np.abs(det) > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        t = np.where(hit, t, -np.inf).max(axis=1)
        out[start : start + 256] = np.where(np.isfinite(t), t, np.nan)
    return out


def world_triangles() -> NDArray[np.float64]:
    mesh = trimesh.load(WORLD_MESH, force="mesh", process=False)
    return mesh.vertices[mesh.faces]


def smooth(values: NDArray[np.float64], passes: int = 6) -> NDArray[np.float64]:
    out = values.copy()
    for _ in range(passes):
        out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
    return out


@dataclass(frozen=True)
class Station:
    """One snip. The tool arrives at ``air`` above the hair, goes straight down to ``low``,
    then straight back up to ``cut``, and closes the blades there. Sideways moves only ever
    happen at ``air``: scissors dragged through hair would snag it."""

    air: Waypoint
    low: Waypoint
    cut: Waypoint
    depth: float  # metres from cut down to low
    rise: float  # metres from cut up to air


def _gaps(tree: cKDTree, rotation: NDArray[np.float64], tcp: NDArray[np.float64]) -> tuple[float, float]:
    """Nearest the scissors and comb, and the fingers, get to the head at this pose."""
    tool = (rotation @ TOOL_POINTS.T).T + tcp
    fingers = (rotation @ FINGER_POINTS.T).T + tcp
    return float(tree.query(tool)[0].min()), float(tree.query(fingers)[0].min())


def scalp_stations(
    tris: NDArray[np.float64],
    tree: cKDTree,
    clearance: float,
    sweep_deg: tuple[float, float] = (-50.0, 58.0),
    row_offsets: tuple[float, ...] = ROW_OFFSETS,
    step: float = 0.025,
    yaw_deg: float = 90.0,
    tilt_scale: float = 0.6,
) -> list[list[Station]]:
    """Snip stations for each row, back of the head to the forehead, ``step`` apart.

    Each row lies in a plane of constant x. Rays from the middle of that slice find the
    scalp; the surface is smoothed so scan bumps don't shake the tool, then pushed out
    along the direction away from the skull centre. That gives where the middle of the
    blades should be when they close; the arm's TCP is placed so the blades land on it.
    The way in and out runs along the tool's own axis, as deep as the real scalp allows.
    """
    a0, a1 = np.radians(sweep_deg)
    rows: list[list[Station]] = []
    for offset in row_offsets:
        origin = SKULL_CENTER + [offset, 0.0, 0.0]
        angles = np.linspace(a0, a1, 200)
        dirs = np.column_stack([np.zeros_like(angles), np.sin(angles), np.cos(angles)])
        reach = ray_hits(tris, origin, dirs)
        if np.isnan(reach).any():
            raise ValueError(f"row at x offset {offset} misses the scalp")
        surface = origin + dirs * smooth(reach)[:, None]
        normals = surface - SKULL_CENTER
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        path = surface + clearance * normals
        # Even spacing along the arc.
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        samples = np.linspace(0.0, arc[-1], max(2, int(np.ceil(arc[-1] / step)) + 1))
        points = np.column_stack([np.interp(samples, arc, path[:, i]) for i in range(3)])
        normal = np.column_stack([np.interp(samples, arc, normals[:, i]) for i in range(3)])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        tangents = np.gradient(points, axis=0)
        row = []
        for point, n, t in zip(points, normal, tangents, strict=True):
            frame = _frame(n, t, np.radians(yaw_deg), tilt_scale)
            tcp = point - frame @ BLADE_CENTER
            into_head = frame[:, 2]
            depth = 0.0
            for d in np.arange(MAX_DEPTH, 0.0, -0.001):
                comb, fingers = _gaps(tree, frame, tcp + d * into_head)
                if comb >= LOW_COMB_GAP and fingers >= LOW_FINGER_GAP:
                    depth = float(d)
                    break
            rise = 0.12
            for r in np.arange(0.01, 0.12, 0.002):
                if min(_gaps(tree, frame, tcp - r * into_head)) >= ABOVE_HAIR:
                    rise = float(r)
                    break
            row.append(
                Station(
                    air=Waypoint(tcp - rise * into_head, frame, cutting=False),
                    low=Waypoint(tcp + depth * into_head, frame, cutting=False),
                    cut=Waypoint(tcp, frame, cutting=True),
                    depth=depth,
                    rise=rise,
                )
            )
        rows.append(row[::-1])  # back of the head first
    return rows


def surface_tree(n: int = 400_000) -> cKDTree:
    """Dense samples of the scan's surface, for distance checks against the real head."""
    mesh = trimesh.load(WORLD_MESH, force="mesh")
    samples, _ = trimesh.sample.sample_surface(mesh, n, seed=2)
    return cKDTree(samples)


def main() -> None:
    mesh = load_scan()
    place = placement(mesh)
    world = to_world(np.asarray(mesh.vertices), place)
    ASSETS.mkdir(exist_ok=True)
    trimesh.Trimesh(world, mesh.faces, process=False).export(WORLD_MESH)
    tris = world[mesh.faces]

    # Scalp map: distance from the skull centre to the surface, by direction.
    lon = (np.arange(MAP_LON) + 0.5) / MAP_LON * 2 * np.pi - np.pi
    lat = (np.arange(MAP_LAT) + 0.5) / MAP_LAT * np.pi - np.pi / 2
    glon, glat = np.meshgrid(lon, lat)
    dirs = np.column_stack(
        [(np.cos(glat) * np.cos(glon)).ravel(), (np.cos(glat) * np.sin(glon)).ravel(), np.sin(glat).ravel()]
    )
    r = ray_hits(tris, SKULL_CENTER, dirs)
    r = np.nan_to_num(r, nan=0.095)
    data = {
        "center": SKULL_CENTER.tolist(),
        "lon": MAP_LON,
        "lat": MAP_LAT,
        "radius_mm10": np.round(r * 1e4).astype(int).tolist(),  # tenths of a millimetre
        **place,
    }
    (PUBLIC_HEAD / "scalp.json").write_text(json.dumps(data, separators=(",", ":")))

    bounds = np.round(world.min(axis=0), 3).tolist(), np.round(world.max(axis=0), 3).tolist()
    print("world bounds", bounds)
    upper = r[dirs[:, 2] > 0.2]
    print(f"skull radius above ears: {upper.min() * 100:.1f}-{upper.max() * 100:.1f} cm")
    tree = surface_tree()
    for clearance in (0.02, 0.03, 0.04):
        rows = scalp_stations(tris, tree, clearance)
        depths = [s.depth * 100 for r in rows for s in r]
        print(f"cut height {clearance}: {[len(r) for r in rows]} stations per row, "
              f"goes {min(depths):.1f}-{max(depths):.1f} cm below it")


if __name__ == "__main__":
    main()
