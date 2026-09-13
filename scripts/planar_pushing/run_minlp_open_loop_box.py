"""Solve one complete rectangular-box pushing plan with a direct MINLP.

Unlike ``run_minlp_mpc_box.py``, this script calls ``solve`` only once.  The
returned trajectory therefore contains the entire contact/free-space schedule
for the requested horizon and is suitable for offline comparisons with the
repository's GCS planner.
"""

from __future__ import annotations

import argparse

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
    MinlpMpcWarmStart,
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


def _rotation_matrix(theta: float) -> np.ndarray:
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


def make_planner(args: argparse.Namespace) -> PlanarBoxPushingMinlpMpc:
    """Build the same box/pusher model as the short MPC demonstration."""

    box = Box2d(width=0.20, height=0.10)
    system = SliderPusherSystemConfig(
        slider=RigidBody("minlp_box", box, mass=0.10),
        pusher_radius=0.015,
        friction_coeff_slider_pusher=0.30,
        friction_coeff_table_slider=0.50,
        integration_constant=0.60,
    )
    config = MinlpMpcConfig(
        horizon=args.horizon,
        time_step=args.time_step,
        position_lower=np.array([-0.40, -0.40]),
        position_upper=np.array([0.40, 0.40]),
        max_pusher_speed=0.60,
        solver_time_limit=args.time_limit,
        success_tolerance=args.success_tolerance,
        required_initial_mode=3 if args.forced_regrasp else None,
    )
    if args.solver == "gurobi":
        return PlanarBoxPushingGurobiMpc(system, config)
    return PlanarBoxPushingMinlpMpc(system, config)


def print_mode_schedule(planner: PlanarBoxPushingMinlpMpc, modes: np.ndarray) -> None:
    """Print consecutive identical modes as one readable schedule block."""

    start = 0
    for end in range(1, len(modes) + 1):
        if end == len(modes) or modes[end] != modes[start]:
            mode_name = planner.mode_name(int(modes[start]))
            print(f"  steps {start:02d}-{end - 1:02d}: {mode_name}")
            start = end


def make_forced_regrasp_warm_start(
    planner: PlanarBoxPushingMinlpMpc, initial_state: PlanarPushingState
) -> MinlpMpcWarmStart:
    """Create a feasible left-face-to-right-face regrasp initialization.

    The first interval retains face 3 contact.  The pusher then travels via
    free regions 3, 0, and 1 before contacting face 1, whose inward normal
    moves the box in the negative body-x direction.  This is a solver-neutral
    warm-start contract, so it works for both BONMIN and native Gurobi.
    """

    h = planner.config.horizon
    if h < 12:
        raise ValueError("The forced-regrasp scenario needs --horizon of at least 12")

    # Body-frame pusher knots.  The top detour is collision-free with the
    # 1.5 cm-radius pusher and respects the 0.6 m/s speed limit at dt = 0.1 s.
    pusher_body_prefix = np.array(
        [
            [-0.115, 0.000],
            [-0.115, 0.000],
            [-0.130, 0.0325],
            [-0.130, 0.0650],
            [-0.078, 0.0650],
            [-0.026, 0.0650],
            [0.026, 0.0650],
            [0.078, 0.0650],
            [0.130, 0.0650],
            [0.130, 0.0325],
            [0.1150, 0.000],
            [0.1150, 0.000],
            [0.1150, 0.000],
        ]
    )
    mode_prefix = np.array([3, 7, 7, 4, 4, 4, 4, 4, 5, 5, 1, 1])
    box_displacement_prefix = np.zeros((12, 2))
    box_displacement_prefix[11] = [-0.02, 0.0]
    box_displacement_prefix = np.vstack(
        (box_displacement_prefix, np.array([[-0.04, 0.0]]))
    )

    pusher_body = np.vstack(
        (pusher_body_prefix, np.repeat(pusher_body_prefix[-1:], h - 12, axis=0))
    )
    box_displacement = np.vstack(
        (
            box_displacement_prefix,
            np.repeat(box_displacement_prefix[-1:], h - 12, axis=0),
        )
    )
    modes = np.concatenate((mode_prefix, np.full(h - 12, 5)))

    theta = initial_state.slider.theta
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    states = np.empty((5, h + 1))
    for k in range(h + 1):
        box_position = (
            initial_state.slider.pos().reshape(-1) + rotation @ box_displacement[k]
        )
        states[0:2, k] = box_position
        states[2, k] = theta
        states[3:5, k] = box_position + rotation @ pusher_body[k]

    contact_force = planner.system.f_max**2 * 0.20
    forces = np.zeros((2, h))
    forces[0, 10:12] = -contact_force
    return MinlpMpcWarmStart(
        states=states,
        pusher_velocities=np.diff(states[3:5], axis=1) / planner.config.time_step,
        contact_forces_B=forces,
        contact_torques_B=np.zeros(h),
        modes=np.eye(8)[:, modes],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("bonmin", "gurobi"), default="bonmin")
    parser.add_argument(
        "--forced-regrasp",
        action="store_true",
        help="Move the box 4 cm in negative body-x; this requires regrasping "
        "from the initial left face to the right face.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Number of fixed-duration contact/free-space intervals.",
    )
    parser.add_argument("--time-step", type=float, default=0.10)
    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="Single-solve time limit in seconds.",
    )
    parser.add_argument("--initial-x", type=float, default=0.0)
    parser.add_argument("--initial-y", type=float, default=0.0)
    parser.add_argument("--initial-theta-deg", type=float, default=0.0)
    parser.add_argument(
        "--pusher-body-x",
        type=float,
        default=-0.115,
        help="Initial pusher x coordinate in the box frame (default: face 3 contact).",
    )
    parser.add_argument("--pusher-body-y", type=float, default=0.0)
    parser.add_argument("--goal-x", type=float)
    parser.add_argument("--goal-y", type=float)
    parser.add_argument("--goal-theta-deg", type=float)
    parser.add_argument("--success-tolerance", type=float, default=2e-3)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save an mp4 of the whole planned trajectory with the repository's "
        "modern (SceneGraph-based) planar visualizer.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trajectories/minlp_open_loop_box",
        help="Filename (without extension) to save the mp4 to, if --visualize is set.",
    )
    args = parser.parse_args()
    args.horizon = args.horizon or (12 if args.forced_regrasp else 4)
    if args.horizon < 1 or args.time_step <= 0.0 or args.time_limit <= 0.0:
        parser.error("--horizon, --time-step, and --time-limit must be positive")
    if args.forced_regrasp and not np.allclose(
        [args.pusher_body_x, args.pusher_body_y], [-0.115, 0.0]
    ):
        parser.error("--forced-regrasp requires the default left-face pusher contact")

    planner = make_planner(args)
    initial_theta = np.deg2rad(args.initial_theta_deg)
    initial_slider = PlanarPose(args.initial_x, args.initial_y, initial_theta)
    initial_rotation = np.array(
        [
            [np.cos(initial_theta), -np.sin(initial_theta)],
            [np.sin(initial_theta), np.cos(initial_theta)],
        ]
    )
    initial_state = PlanarPushingState(
        slider=initial_slider,
        # The default is the centre of face 3, one pusher radius outside the
        # box.  Expressing it in the box frame keeps that contact valid when
        # --initial-x/--initial-y/--initial-theta-deg are changed.
        pusher_position=initial_slider.pos().reshape(-1)
        + initial_rotation @ np.array([args.pusher_body_x, args.pusher_body_y]),
    )
    if args.forced_regrasp:
        target_position = initial_slider.pos().reshape(-1) + initial_rotation @ np.array(
            [-0.04, 0.0]
        )
        goal = PlanarPushingGoal(PlanarPose(*target_position, initial_theta))
        warm_start = make_forced_regrasp_warm_start(planner, initial_state)
        print("Scenario: forced regrasp (initial contact face 3 -> contact face 1)")
    else:
        goal = PlanarPushingGoal(
            PlanarPose(
                args.goal_x if args.goal_x is not None else 0.08,
                args.goal_y if args.goal_y is not None else 0.0,
                np.deg2rad(
                    args.goal_theta_deg if args.goal_theta_deg is not None else 0.0
                ),
            )
        )
        warm_start = None

    try:
        solution = planner.solve(initial_state, goal, warm_start=warm_start)
    except MinlpSolveError as exc:
        raise SystemExit(f"{args.solver} open-loop solve failed: {exc}") from exc

    terminal = PlanarPushingState.from_vector(solution.states[:, -1])
    position_error = np.linalg.norm(terminal.slider.pos() - goal.slider.pos())
    orientation_error = np.arctan2(
        np.sin(terminal.slider.theta - goal.slider.theta),
        np.cos(terminal.slider.theta - goal.slider.theta),
    )
    print(f"Solver: {solution.solver_stats.get('solver', args.solver)}")
    print(f"Objective: {solution.objective:.5g}")
    print(f"Terminal slider pose: {terminal.slider}")
    print(
        "Terminal errors: "
        f"position={position_error:.4g} m, "
        f"orientation={np.rad2deg(orientation_error):.3f} deg"
    )
    print(f"Goal reached: {planner.goal_reached(terminal, goal)}")
    print("Mode schedule:")
    print_mode_schedule(planner, solution.mode_indices)

    if args.visualize:
        states = solution.states
        p_WBs = states[0:2].T
        R_WBs = [_rotation_matrix(theta) for theta in states[2]]
        p_WPs = states[3:5].T
        # contact_forces_B is already in physical Newtons: force_scale only
        # sizes the optimization bound, it is not a further rescaling of the
        # solved decision variable.
        f_c_Ws = np.column_stack(
            [
                R_WBs[k] @ solution.contact_forces_B[:, k]
                for k in range(solution.contact_forces_B.shape[1])
            ]
            + [np.zeros(2)]
        ).T
        plan_config = PlanarPlanConfig(
            dynamics_config=planner.system,
            start_and_goal=PlanarPushingStartAndGoal(
                slider_initial_pose=initial_state.slider,
                slider_target_pose=goal.slider,
                pusher_initial_pose=PlanarPose(*initial_state.pusher_position, 0.0),
                pusher_target_pose=PlanarPose(*terminal.pusher_position, 0.0),
            ),
        )
        trajectory = SimplePlanarPushingTrajectory(
            p_WBs, R_WBs, p_WPs, f_c_Ws, args.time_step, plan_config
        )
        visualize_planar_pushing_trajectory(
            trajectory, save=True, show=False, filename=args.output
        )
        print(f"Saved animation to {args.output}.mp4")


if __name__ == "__main__":
    main()
