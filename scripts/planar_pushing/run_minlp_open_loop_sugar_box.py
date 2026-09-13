"""Open-loop direct-MINLP plan for the exact scenario used by create_plans.py.

Parameters below are pulled directly from
trajectories/run_20260913150534_sugar_box/traj_0/trajectory/traj_rounded.pkl,
the trajectory that `python scripts/planar_pushing/create_plans.py --body
sugar_box --seed 0 --num 1` produced with the GCS planner. That GCS solution's
path visits: NON_COLL_3 (approach) -> NON_COLL_0 -> NON_COLL_1 -> CONTACT_1
(push) -> NON_COLL_1 -> NON_COLL_0 -> CONTACT_0 (push) -> NON_COLL_0 ->
NON_COLL_3 (approach to final pose) -- i.e. two separate contact faces.

This script asks the same question the box experiments did: does the direct
open-loop MINLP (solved once over a long horizon, not receding-horizon MPC)
find a comparable multi-face plan for the *same* box geometry, dynamics
parameters, and start/goal poses that the repository's own GCS planner used?

Unlike the box experiments, this box is taller than it is wide (0.106 x
0.185m) and the task needs ~37 degrees of rotation plus translation, so it is
a materially harder search than the earlier forced-regrasp box demo.
"""

import argparse
import time

import numpy as np

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    SimplePlanarPushingTrajectory,
)
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
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    SliderPusherSystemConfig,
)
from planning_through_contact.visualize.planar_pushing import (
    visualize_planar_pushing_trajectory,
)

# Exact parameters recovered from the reference GCS pickle (see module docstring).
SUGAR_BOX_WIDTH = 0.106
SUGAR_BOX_HEIGHT = 0.185
PUSHER_RADIUS = 0.015
SLIDER_MASS = 0.1
FRICTION_COEFF_TABLE_SLIDER = 0.5
FRICTION_COEFF_SLIDER_PUSHER = 0.1
INTEGRATION_CONSTANT = 0.3

INITIAL_SLIDER_POSE = PlanarPose(0.02928810235639484, 0.1291136198234517, 0.64362606712809)
TARGET_SLIDER_POSE = PlanarPose(0.0, 0.0, 0.0)


def rotation_matrix(theta: float) -> np.ndarray:
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


def make_initial_pusher_position() -> np.ndarray:
    """A free-space start point just outside face 3, matching where the
    reference GCS path also begins its approach (NON_COLL_3)."""
    box = Box2d(width=SUGAR_BOX_WIDTH, height=SUGAR_BOX_HEIGHT)
    v3, v0 = box.get_proximate_vertices_from_location(box.contact_locations[3])
    face_3_midpoint_B = ((v3 + v0) / 2).flatten()
    clearance = PUSHER_RADIUS + 0.035
    p_BP_free = face_3_midpoint_B + np.array([-clearance, 0.0])
    R_WB = rotation_matrix(INITIAL_SLIDER_POSE.theta)
    return INITIAL_SLIDER_POSE.pos().flatten() + R_WB @ p_BP_free


def print_mode_schedule(planner: PlanarBoxPushingMinlpMpc, modes: np.ndarray) -> None:
    start = 0
    for end in range(1, len(modes) + 1):
        if end == len(modes) or modes[end] != modes[start]:
            print(f"  steps {start:02d}-{end - 1:02d}: {planner.mode_name(int(modes[start]))}")
            start = end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("bonmin", "gurobi"), default="gurobi")
    parser.add_argument("--horizon", type=int, default=22)
    parser.add_argument("--time-step", type=float, default=0.10)
    parser.add_argument("--time-limit", type=float, default=400.0)
    parser.add_argument("--success-tolerance", type=float, default=0.015)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--output", type=str, default="trajectories/minlp_open_loop_sugar_box"
    )
    args = parser.parse_args()

    box = Box2d(width=SUGAR_BOX_WIDTH, height=SUGAR_BOX_HEIGHT)
    system = SliderPusherSystemConfig(
        slider=RigidBody("sugar_box", box, mass=SLIDER_MASS),
        pusher_radius=PUSHER_RADIUS,
        friction_coeff_slider_pusher=FRICTION_COEFF_SLIDER_PUSHER,
        friction_coeff_table_slider=FRICTION_COEFF_TABLE_SLIDER,
        integration_constant=INTEGRATION_CONSTANT,
    )
    config = MinlpMpcConfig(
        horizon=args.horizon,
        time_step=args.time_step,
        position_lower=np.array([-0.5, -0.5]),
        position_upper=np.array([0.5, 0.5]),
        max_pusher_speed=0.6,
        solver_time_limit=args.time_limit,
        success_tolerance=args.success_tolerance,
    )
    planner_cls = PlanarBoxPushingGurobiMpc if args.solver == "gurobi" else PlanarBoxPushingMinlpMpc
    planner = planner_cls(system, config)

    initial_state = PlanarPushingState(
        slider=INITIAL_SLIDER_POSE,
        pusher_position=make_initial_pusher_position(),
    )
    goal = PlanarPushingGoal(TARGET_SLIDER_POSE)

    print(f"Solver: {args.solver}, horizon={args.horizon}, time_limit={args.time_limit}s")
    print(f"Start: {initial_state.slider}, pusher at {initial_state.pusher_position}")
    print(f"Goal:  {goal.slider}")

    t0 = time.time()
    try:
        solution = planner.solve(initial_state, goal)
    except MinlpSolveError as exc:
        raise SystemExit(f"Open-loop solve failed: {exc}") from exc
    elapsed = time.time() - t0

    terminal = PlanarPushingState.from_vector(solution.states[:, -1])
    position_error = np.linalg.norm(terminal.slider.pos() - goal.slider.pos())
    orientation_error = np.arctan2(
        np.sin(terminal.slider.theta - goal.slider.theta),
        np.cos(terminal.slider.theta - goal.slider.theta),
    )
    print(f"\nSolved in {elapsed:.1f}s")
    print(f"Objective: {solution.objective:.5g}")
    print(f"Terminal slider pose: {terminal.slider}")
    print(
        f"Terminal errors: position={position_error:.4g} m, "
        f"orientation={np.rad2deg(orientation_error):.3f} deg"
    )
    print(f"Goal reached: {planner.goal_reached(terminal, goal)}")
    print("Mode schedule:")
    print_mode_schedule(planner, solution.mode_indices)
    faces_used = sorted({int(m) for m in solution.mode_indices if m < 4})
    print(f"Distinct contact faces used: {faces_used}")

    if args.visualize:
        states = solution.states
        p_WBs = states[0:2].T
        R_WBs = [rotation_matrix(theta) for theta in states[2]]
        p_WPs = states[3:5].T
        f_c_Ws = np.column_stack(
            [
                R_WBs[k] @ solution.contact_forces_B[:, k]
                for k in range(solution.contact_forces_B.shape[1])
            ]
            + [np.zeros(2)]
        ).T
        plan_config = PlanarPlanConfig(
            dynamics_config=system,
            start_and_goal=PlanarPushingStartAndGoal(
                slider_initial_pose=initial_state.slider,
                slider_target_pose=goal.slider,
                pusher_initial_pose=PlanarPose(*initial_state.pusher_position, 0.0),
                pusher_target_pose=PlanarPose(*terminal.pusher_position, 0.0),
            ),
        )
        trajectory = SimplePlanarPushingTrajectory(
            p_WBs, R_WBs, p_WPs, f_c_Ws, config.time_step, plan_config
        )
        visualize_planar_pushing_trajectory(
            trajectory, save=True, show=False, filename=args.output
        )
        print(f"Saved animation to {args.output}.mp4")


if __name__ == "__main__":
    main()
