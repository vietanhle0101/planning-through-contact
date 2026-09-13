"""Run the first rectangular-box MINLP/MPC example.

The pusher starts on the left face and drives the box 4 cm to the right.  The
default execution is model-consistent MPC; pass ``--visualize`` to replay its
executed first steps with the repository's existing legacy planar visualizer.
"""

import argparse

import numpy as np

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.gurobi_minlp_mpc import (
    PlanarBoxPushingGurobiMpc,
)
from planning_through_contact.planning.planar.minlp_mpc import (
    MinlpMpcConfig,
    MinlpSolveError,
    PlanarBoxPushingMinlpMpc,
    PlanarPushingGoal,
    PlanarPushingState,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    SliderPusherSystemConfig,
)
from planning_through_contact.visualize.planar_pushing import (
    visualize_planar_pushing_trajectory_legacy,
)


def make_example_planner(solver: str) -> PlanarBoxPushingMinlpMpc:
    box = Box2d(width=0.20, height=0.10)
    system = SliderPusherSystemConfig(
        slider=RigidBody("minlp_box", box, mass=0.10),
        pusher_radius=0.015,
        friction_coeff_slider_pusher=0.30,
        friction_coeff_table_slider=0.50,
        integration_constant=0.60,
    )
    mpc_config = MinlpMpcConfig(
        horizon=2,
        time_step=0.10,
        position_lower=np.array([-0.40, -0.40]),
        position_upper=np.array([0.40, 0.40]),
        success_tolerance=2e-3,
    )
    if solver == "gurobi":
        return PlanarBoxPushingGurobiMpc(system, mpc_config)
    return PlanarBoxPushingMinlpMpc(system, mpc_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Replay executed MPC first steps using the existing planar visualizer.",
    )
    parser.add_argument(
        "--solver",
        choices=("bonmin", "gurobi"),
        default="bonmin",
        help="MINLP backend. Gurobi requires a valid local Gurobi license.",
    )
    args = parser.parse_args()

    planner = make_example_planner(args.solver)
    state = PlanarPushingState(
        slider=PlanarPose(0.0, 0.0, 0.0),
        # Face 3 is the left face.  Its center is x = -width/2, and the
        # cylindrical pusher center is one pusher radius farther left.
        pusher_position=np.array([-0.115, 0.0]),
    )
    goal = PlanarPushingGoal(PlanarPose(0.04, 0.0, 0.0))
    try:
        history = planner.run(state, goal, max_iterations=6)
    except MinlpSolveError as exc:
        raise SystemExit(f"{args.solver} MPC solve failed: {exc}") from exc
    if not history:
        print("Goal was already satisfied.")
        return

    final_state = history[-1].predicted_next_state
    print(f"Executed {len(history)} MPC iterations")
    print(f"Final slider pose: {final_state.slider}")
    print("Executed modes:", [planner.mode_name(step.first_mode) for step in history])

    if args.visualize:
        trajectory = planner.to_legacy_trajectory(history)
        visualize_planar_pushing_trajectory_legacy(
            trajectory,
            planner.geometry,
            planner.system.pusher_radius,
        )


if __name__ == "__main__":
    main()
