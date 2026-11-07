"""Clipper-style tool path over a spherical head.

Pure geometry (numpy only). The head is a sphere; the tool rides on a second,
larger sphere ``clearance`` metres off the scalp. Each pass is the arc where a
vertical plane cuts that offset sphere, so passes are parallel rows like a real
clipper, ``row_spacing`` apart. Every waypoint's tool z-axis points into the
head and its x-axis points along the stroke. With ``tilt_scale`` 1 the tool is
normal to the scalp; smaller values tilt it only part of the way from vertical,
which keeps the wrist in reach on the sides of the head. The tool tip stays
``clearance`` off the scalp either way.

Strokes all run the same way, like a clipper. Each row is returned on its own:
lower onto the head along the normal from ``lift`` metres out, stroke, lift off.
Moving between rows is left to a collision-checked joint-space planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Vec = NDArray[np.float64]
Mat = NDArray[np.float64]


@dataclass(frozen=True)
class Waypoint:
    position: Vec  # world xyz, metres
    rotation: Mat  # 3x3, columns are the tool's x, y, z axes in world frame
    cutting: bool  # True while the clipper is on a stroke


def _tilted(normal: Vec, tilt_scale: float) -> Vec:
    """Rotate world-up toward ``normal`` by ``tilt_scale`` of the angle between them."""
    up = np.array([0.0, 0.0, 1.0])
    n = normal / np.linalg.norm(normal)
    axis = np.cross(up, n)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return n
    axis /= s
    angle = tilt_scale * np.arctan2(s, np.dot(up, n))
    return up * np.cos(angle) + np.cross(axis, up) * np.sin(angle) + axis * np.dot(axis, up) * (
        1 - np.cos(angle)
    )


def _frame(normal: Vec, tangent: Vec, yaw: float = 0.0, tilt_scale: float = 1.0) -> Mat:
    """Tool frame: z into the head, x along the stroke turned ``yaw`` rad about z."""
    z = -_tilted(normal, tilt_scale)
    x = tangent - np.dot(tangent, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    x, y = np.cos(yaw) * x + np.sin(yaw) * y, -np.sin(yaw) * x + np.cos(yaw) * y
    return np.column_stack([x, y, z])


def _arc(
    center: Vec, radius: float, offset: float, angles: Vec, along: Literal["x", "y"]
) -> tuple[list[Vec], list[Vec]]:
    """Points and unit outward normals on the row at lateral ``offset``."""
    r = np.sqrt(radius**2 - offset**2)
    points, tangents = [], []
    for a in angles:
        s, c = np.sin(a), np.cos(a)
        if along == "x":
            local, tangent = np.array([r * s, offset, r * c]), np.array([c, 0.0, -s])
        else:
            local, tangent = np.array([offset, r * s, r * c]), np.array([0.0, c, -s])
        points.append(center + local)
        tangents.append(tangent)
    return points, tangents


def clipper_rows(
    center: Vec,
    head_radius: float = 0.10,
    clearance: float = 0.02,
    stroke_along: Literal["x", "y"] = "x",
    sweep_deg: tuple[float, float] = (-55.0, 55.0),
    row_spacing: float = 0.04,
    n_rows: int = 5,
    step: float = 0.015,
    lift: float = 0.05,
    yaw_deg: float = 0.0,
    tilt_scale: float = 1.0,
) -> list[list[Waypoint]]:
    """One waypoint list per parallel clipper stroke over the top of the head."""
    center = np.asarray(center, dtype=np.float64)
    radius = head_radius + clearance
    offsets = (np.arange(n_rows) - (n_rows - 1) / 2) * row_spacing
    if np.max(np.abs(offsets)) >= radius:
        raise ValueError("rows extend past the side of the head")

    a0, a1 = np.radians(sweep_deg)
    rows: list[list[Waypoint]] = []
    for offset in offsets:
        r_row = np.sqrt(radius**2 - offset**2)
        n = max(2, int(np.ceil(r_row * (a1 - a0) / step)) + 1)
        points, tangents = _arc(center, radius, offset, np.linspace(a0, a1, n), stroke_along)
        frames = [
            _frame(p - center, t, np.radians(yaw_deg), tilt_scale)
            for p, t in zip(points, tangents, strict=True)
        ]
        normals = [(p - center) / radius for p in points]

        # Lowered onto the head along the normal, stroke, then lift off.
        rows.append(
            [
                Waypoint(points[0] + lift * normals[0], frames[0], cutting=False),
                *(Waypoint(p, f, cutting=True) for p, f in zip(points, frames, strict=True)),
                Waypoint(points[-1] + lift * normals[-1], frames[-1], cutting=False),
            ]
        )
    return rows


def min_clearance(path: list[Waypoint], center: Vec, head_radius: float) -> float:
    """Smallest distance from any waypoint, or any straight segment between them, to the scalp."""
    center = np.asarray(center, dtype=np.float64)
    best = np.inf
    for a, b in zip(path, path[1:], strict=False):
        ab = b.position - a.position
        t = np.clip(np.dot(center - a.position, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
        closest = a.position + t * ab
        best = min(best, float(np.linalg.norm(closest - center)) - head_radius)
    return best
