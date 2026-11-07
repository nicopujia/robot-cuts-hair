"""ManipulationModule with one extra RPC for multi-waypoint Cartesian paths.

DimOS keeps Cartesian path planning internal (``generate_cartesian_plan``):
no skill, MCP tool, or CLI command exposes it, and the only Cartesian RPC is
``move_linear`` (a single straight segment). This subclass exposes that internal
method as ``plan_cartesian_waypoints``. Nothing in DimOS itself is modified.

Kept in its own file so clients can import the class (``Arm.from_app`` needs it).
"""

from __future__ import annotations

from dimos.core.core import rpc
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.manipulation_spec import PlanResult, PlanStatus
from dimos.manipulation.planning.planners.config import CartesianPathConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped


class HaircutManipulationModule(ManipulationModule):
    @rpc
    def plan_cartesian_waypoints(
        self,
        group_id: str,
        waypoints: list[PoseStamped],
        max_linear_speed: float = 0.05,
        blend_deviation: float = 0.005,
        check_collision: bool = True,
        planner_overrides: dict[str, float] | None = None,
    ) -> PlanResult:
        """Plan a TCP path through world-frame waypoints; the first must be the current TCP pose.

        ``planner_overrides`` sets other RoboPlanCartesianPathConfig fields by name.
        """
        if len(waypoints) < 2:
            return PlanResult(PlanStatus.INVALID_TARGET, "Need at least two waypoints")
        config = CartesianPathConfig(
            speed_mode="bounded",
            max_linear_speed=max_linear_speed,
            toppra_blend_deviation=blend_deviation,
            **(planner_overrides or {}),
        )
        plan = self.generate_cartesian_plan(
            {group_id: list(waypoints)}, config, check_collision=check_collision
        )
        if plan is None:
            return self._planning_failure()
        return PlanResult(PlanStatus.SUCCEEDED, plan.message, plan)

    @rpc
    def set_camera(self, position: list[float], look_at: list[float]) -> bool:
        """Point the Viser camera (connected and future clients) for screen recording.

        Cosmetic only. Reaches into private attributes, since DimOS exposes no camera API.
        """
        visualization = getattr(self._world_monitor, "_visualization", None)
        server = getattr(visualization, "_server", None)
        if server is None:
            return False
        server.initial_camera.position = tuple(position)
        server.initial_camera.look_at = tuple(look_at)
        for client in server.get_clients().values():
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(look_at)
        return True
