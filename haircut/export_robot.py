"""Export the xArm7 used by DimOS for the web demo: kinematics JSON + decimated GLB meshes.

Run in two stages because mesh decimation needs a package DimOS doesn't ship:
    ./py export_robot.py dump /tmp/xarm7.urdf
    uv run --no-project --python 3.12 --with trimesh --with fast-simplification --with numpy \
        python export_robot.py build /tmp/xarm7.urdf ../web/public/robot
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def dump(out: Path) -> None:
    from dimos.robot.manipulators.xarm.config import make_xarm7_model_config

    loaded = make_xarm7_model_config(add_gripper=True, gripper_hardware_id="arm").model.load()
    pkg = loaded.package_paths["xarm_description"]
    xml = loaded.xml if isinstance(loaded.xml, str) else ET.tostring(loaded.xml, encoding="unicode")
    out.write_text(xml.replace("package://xarm_description", str(pkg)))
    print("wrote", out, "license:", (Path(pkg).parent / "LICENSE").exists())


def vec(text: str | None, default: list[float]) -> list[float]:
    return [float(v) for v in text.split()] if text else default


def build(urdf: Path, out: Path) -> None:
    import numpy as np
    import trimesh
    from trimesh.transformations import euler_matrix

    root = ET.parse(urdf).getroot()
    out.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    total = 0
    for link in root.findall("link"):
        parts = []
        color = None
        for visual in link.findall("visual"):
            mesh_el = visual.find("geometry/mesh")
            if mesh_el is None:
                continue
            path = mesh_el.get("filename", "").removeprefix("file://")
            mesh = trimesh.load(path, force="mesh")
            mesh.apply_scale(vec(mesh_el.get("scale"), [1.0, 1.0, 1.0]))
            origin = visual.find("origin")
            rpy = vec(origin.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0])
            xyz = vec(origin.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0])
            matrix = euler_matrix(*rpy, axes="sxyz")
            matrix[:3, 3] = xyz
            mesh.apply_transform(matrix)
            parts.append(mesh)
            rgba = visual.find("material/color")
            color = vec(rgba.get("rgba"), [0.8, 0.8, 0.8, 1.0]) if rgba is not None else color
        if not parts:
            continue
        mesh = trimesh.util.concatenate(parts)
        mesh.merge_vertices()
        if len(mesh.faces) > 6000:
            mesh = mesh.simplify_quadric_decimation(face_count=6000)
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh, face_colors=np.array((color or [0.8, 0.8, 0.82, 1.0])) * 255
        )
        total += len(mesh.faces)
        scene.add_geometry(mesh, node_name=link.get("name"), geom_name=link.get("name"))
    scene.export(out / "xarm7.glb")

    joints = []
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        mimic = joint.find("mimic")
        limit = joint.find("limit")
        joints.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "xyz": vec(origin.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0]),
                "rpy": vec(origin.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0]),
                "axis": vec(axis.get("xyz") if axis is not None else None, [0.0, 0.0, 1.0]),
                "mimic": None
                if mimic is None
                else {
                    "joint": mimic.get("joint"),
                    "multiplier": float(mimic.get("multiplier", 1)),
                    "offset": float(mimic.get("offset", 0)),
                },
                "upper": float(limit.get("upper", 0)) if limit is not None else None,
                "lower": float(limit.get("lower", 0)) if limit is not None else None,
                "effort": float(limit.get("effort", 0)) if limit is not None else None,
                "velocity": float(limit.get("velocity", 0)) if limit is not None else None,
            }
        )
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            links[link.get("name")] = {
                "mass": float(inertial.find("mass").get("value")),
                "com": vec(origin.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0]),
            }
    (out / "kinematics.json").write_text(json.dumps({"joints": joints, "links": links}))
    size = (out / "xarm7.glb").stat().st_size / 1e6
    print(f"{len(scene.geometry)} link meshes, {total} faces, {size:.2f} MB, {len(joints)} joints")
    print([j["name"] for j in joints if j["type"] != "fixed"])
    print([j["name"] for j in joints if j["type"] == "fixed"][:12])


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(Path(sys.argv[2]))
    else:
        build(Path(sys.argv[2]), Path(sys.argv[3]))
