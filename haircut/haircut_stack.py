"""Mock xArm7 planning stack using HaircutManipulationModule.

Mirrors the stock ``xarm7-planner-coordinator`` blueprint (mock arm when no
address is given) with Viser visualization switched on.

Run from the dimos checkout, so its .venv is used:
    PYTHONPATH=../haircut .venv/bin/python ../haircut/haircut_stack.py
"""

from __future__ import annotations

from haircut_module import HaircutManipulationModule

from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.robot.manipulators.common.blueprints import coordinator, trajectory_task
from dimos.robot.manipulators.xarm.config import make_xarm7_model_config, xarm7_hardware

_xarm7_hw = xarm7_hardware("arm", gripper=True, mock_without_address=True)

haircut_stack = autoconnect(
    HaircutManipulationModule.blueprint(
        model=make_xarm7_model_config(add_gripper=True, gripper_hardware_id="arm"),
        planning_timeout=10.0,
        visualization={"backend": "viser"},
        # Default is 200; tilted poses at the edge of the head need more to converge.
        kinematics={"backend": "pink", "max_iterations": 1000},
    ),
    coordinator(
        hardware=[_xarm7_hw],
        tasks=[
            trajectory_task(_xarm7_hw),
            TaskConfig(name="arm_gripper", type="gripper", joint_names=["arm/gripper"], priority=20),
        ],
    ),
)

if __name__ == "__main__":
    ModuleCoordinator.build(haircut_stack).loop()
