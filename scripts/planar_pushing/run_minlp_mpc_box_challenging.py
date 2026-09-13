"""A harder rectangular-box MINLP/MPC example than run_minlp_mpc_box.py.

The simple demo only pushes the box 4 cm along the axis the initial contact
face already points at, with a 2-step horizon and 2 MPC iterations. This
script instead moves the box diagonally and rotates it by ~30 degrees --
enough displacement in the direction *orthogonal* to the initial contact
face's normal that a single fixed sticking contact is unlikely to get there
efficiently through friction alone. That gives the planner a real reason to
release contact, reposition the pusher in free space (regrasp), and push
again from a different face, exercising the mode-switching machinery that
the simple demo never has to touch.

This script does not modify minlp_mpc.py, run_minlp_mpc_box.py, or their
tests. It only builds on the existing public API.
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
from planning_through_contact.planning.planar.minlp_mpc import (
    MinlpMpcConfig,
    MinlpMpcSolution,
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


def make_challenging_planner() -> PlanarBoxPushingMinlpMpc:
    box = Box2d(width=0.20, height=0.10)
    system = SliderPusherSystemConfig(
        slider=RigidBody("minlp_box", box, mass=0.10),
        pusher_radius=0.015,
        friction_coeff_slider_pusher=0.30,
        friction_coeff_table_slider=0.50,
        integration_constant=0.60,
    )
    mpc_config = MinlpMpcConfig(
        horizon=3,
        time_step=0.10,
        position_lower=np.array([-0.50, -0.50]),
        position_upper=np.array([0.50, 0.50]),
        max_pusher_speed=0.6,
        solver_time_limit=8.0,
        success_tolerance=1e-2,
    )
    return PlanarBoxPushingMinlpMpc(system, mpc_config)


def rotation_matrix(theta: float) -> np.ndarray:
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


def history_to_trajectory(
    history: list[MinlpMpcSolution],
    planner: PlanarBoxPushingMinlpMpc,
    initial_state: PlanarPushingState,
    goal: PlanarPushingGoal,
) -> SimplePlanarPushingTrajectory:
    """Builds a repo-native trajectory from the executed MPC steps.

    Only the executed first step of each solve is kept (not the rest of the
    predicted horizon), mirroring what a real receding-horizon controller
    would have actually done.
    """
    states = np.column_stack(
        [initial_state.vector()] + [sol.states[:, 1] for sol in history]
    )
    p_WBs = states[0:2].T
    thetas = states[2]
    R_WBs = [rotation_matrix(theta) for theta in thetas]
    p_WPs = states[3:5].T
    # contact_forces_B is already in physical Newtons: force_scale only sizes
    # the optimization bound (see MinlpMpcConfig.max_normal_force), it is not
    # a further rescaling of the solved decision variable.
    f_c_Ws = np.column_stack(
        [R_WBs[k] @ sol.contact_forces_B[:, 0] for k, sol in enumerate(history)]
        + [np.zeros(2)]
    ).T

    plan_config = PlanarPlanConfig(
        dynamics_config=planner.system,
        start_and_goal=PlanarPushingStartAndGoal(
            slider_initial_pose=initial_state.slider,
            slider_target_pose=goal.slider,
            pusher_initial_pose=PlanarPose(
                initial_state.pusher_position[0],
                initial_state.pusher_position[1],
                0.0,
            ),
            pusher_target_pose=PlanarPose(
                history[-1].predicted_next_state.pusher_position[0],
                history[-1].predicted_next_state.pusher_position[1],
                0.0,
            ),
        ),
    )

    return SimplePlanarPushingTrajectory(
        p_WBs, R_WBs, p_WPs, f_c_Ws, planner.config.time_step, plan_config
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save an mp4 of the executed MPC steps with the repository's "
        "modern (SceneGraph-based) planar visualizer.",
    )
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument(
        "--output",
        type=str,
        default="trajectories/minlp_mpc_challenging",
        help="Filename (without extension) to save the mp4 to, if --visualize is set.",
    )
    args = parser.parse_args()

    planner = make_challenging_planner()

    # Start in contact with the left face (face 3), same convention as
    # run_minlp_mpc_box.py: the face center is at x = -width/2, and the
    # cylindrical pusher center sits one pusher radius further out.
    initial_state = PlanarPushingState(
        slider=PlanarPose(0.0, 0.0, 0.0),
        pusher_position=np.array([-0.115, 0.0]),
    )
    # 15 cm in x, 8 cm in y, and a 30 degree rotation: the y-displacement is
    # orthogonal to what face 3 alone can efficiently produce (it can only
    # move the box sideways via friction, bounded by mu=0.30), so a good
    # solution likely needs to release, reposition, and push again from a
    # different face.
    goal = PlanarPushingGoal(PlanarPose(0.15, 0.08, np.deg2rad(30)))

    print(f"Start: {initial_state.slider}")
    print(f"Goal:  {goal.slider}")

    history: list[MinlpMpcSolution] = []
    state = initial_state
    warm_start = None
    previous_mode = None
    start_time = time.time()
    for iteration in range(args.max_iterations):
        if planner.goal_reached(state, goal):
            break
        try:
            solution = planner.solve(
                state, goal, previous_mode=previous_mode, warm_start=warm_start
            )
        except MinlpSolveError as exc:
            print(f"Iteration {iteration}: solve failed ({exc}); stopping.")
            break
        history.append(solution)
        mode_name = planner.mode_name(solution.first_mode)
        next_state = solution.predicted_next_state
        pos_error = np.linalg.norm(next_state.slider.pos() - goal.slider.pos())
        print(
            f"Iter {iteration:2d}: mode={mode_name:<16s} "
            f"pose=({next_state.slider.x:+.3f}, {next_state.slider.y:+.3f}, "
            f"{np.rad2deg(next_state.slider.theta):+.1f} deg)  "
            f"pos_err={pos_error * 100:.2f} cm"
        )
        state = next_state
        warm_start = solution.warm_start.shifted()
        previous_mode = solution.first_mode

    elapsed = time.time() - start_time
    if not history:
        print("Goal was already satisfied, or the very first solve failed.")
        return

    final_state = history[-1].predicted_next_state
    print(f"\nExecuted {len(history)} MPC iterations in {elapsed:.1f}s")
    print(f"Final slider pose: {final_state.slider}")
    reached = planner.goal_reached(final_state, goal)
    print(f"Goal reached (tol={planner.config.success_tolerance}): {reached}")
    print(
        "Mode sequence:",
        [planner.mode_name(step.first_mode) for step in history],
    )

    if args.visualize:
        trajectory = history_to_trajectory(history, planner, initial_state, goal)
        visualize_planar_pushing_trajectory(
            trajectory, save=True, show=False, filename=args.output
        )
        print(f"Saved animation to {args.output}.mp4")


if __name__ == "__main__":
    main()
