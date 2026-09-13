"""MINLP/MPC example that mirrors an existing GCS-planned sugar-box run.

The start/goal/geometry/dynamics parameters here are pulled directly from
trajectories/run_20260913150534_sugar_box/traj_0/trajectory/traj_rounded.pkl,
a plan the GCS planner produced whose path visits FACE_3 (free space) ->
FACE_0 (free space) -> FACE_1 (free space) -> FACE_1 (contact) -> FACE_1
(free space) -> FACE_0 (free space) -> FACE_0 (contact) -> FACE_0 (free
space) -> FACE_3 (free space): i.e. it approaches, pushes on face 1, releases,
regrasps, and pushes again on face 0.

This script asks the direct-MINLP MPC planner (from the other, already
reviewed minlp_mpc.py) to solve the same problem -- same box, same starting
pose, same target pose, same friction/mass parameters -- to see whether it
also discovers a multi-face solution, and to compare against the GCS
reference qualitatively. It does not modify minlp_mpc.py or the GCS planner.
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


def make_planner() -> PlanarBoxPushingMinlpMpc:
    box = Box2d(width=SUGAR_BOX_WIDTH, height=SUGAR_BOX_HEIGHT)
    system = SliderPusherSystemConfig(
        slider=RigidBody("sugar_box", box, mass=SLIDER_MASS),
        pusher_radius=PUSHER_RADIUS,
        friction_coeff_slider_pusher=FRICTION_COEFF_SLIDER_PUSHER,
        friction_coeff_table_slider=FRICTION_COEFF_TABLE_SLIDER,
        integration_constant=INTEGRATION_CONSTANT,
    )
    mpc_config = MinlpMpcConfig(
        horizon=3,
        time_step=0.10,
        position_lower=np.array([-0.40, -0.40]),
        position_upper=np.array([0.40, 0.40]),
        max_pusher_speed=0.6,
        solver_time_limit=10.0,
        success_tolerance=1.5e-2,
    )
    return PlanarBoxPushingMinlpMpc(system, mpc_config)


def make_initial_pusher_position() -> np.ndarray:
    """A free-space start point just outside the box's face 3 (the same face
    the reference GCS path starts its approach from), so the MPC has to
    approach and make first contact itself, like the GCS plan does."""
    box = Box2d(width=SUGAR_BOX_WIDTH, height=SUGAR_BOX_HEIGHT)
    v3, v0 = box.get_proximate_vertices_from_location(
        box.contact_locations[3]
    )
    face_3_midpoint_B = ((v3 + v0) / 2).flatten()
    clearance = PUSHER_RADIUS + 0.035
    p_BP_free = face_3_midpoint_B + np.array([-clearance, 0.0])
    R_WB = rotation_matrix(INITIAL_SLIDER_POSE.theta)
    p_WP = INITIAL_SLIDER_POSE.pos().flatten() + R_WB @ p_BP_free
    return p_WP


def history_to_trajectory(
    history: list[MinlpMpcSolution],
    planner: PlanarBoxPushingMinlpMpc,
    initial_state: PlanarPushingState,
    goal: PlanarPushingGoal,
) -> SimplePlanarPushingTrajectory:
    states = np.column_stack(
        [initial_state.vector()] + [sol.states[:, 1] for sol in history]
    )
    p_WBs = states[0:2].T
    thetas = states[2]
    R_WBs = [rotation_matrix(theta) for theta in thetas]
    p_WPs = states[3:5].T
    # contact_forces_B is already in physical Newtons; force_scale only sizes
    # the optimization bound, it does not further rescale the solution.
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
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument(
        "--output", type=str, default="trajectories/minlp_mpc_sugar_box_regrasp"
    )
    args = parser.parse_args()

    planner = make_planner()
    initial_state = PlanarPushingState(
        slider=INITIAL_SLIDER_POSE,
        pusher_position=make_initial_pusher_position(),
    )
    goal = PlanarPushingGoal(TARGET_SLIDER_POSE)

    print(f"Start: {initial_state.slider}, pusher at {initial_state.pusher_position}")
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
    print(
        f"Goal reached (tol={planner.config.success_tolerance}): "
        f"{planner.goal_reached(final_state, goal)}"
    )
    modes = [planner.mode_name(step.first_mode) for step in history]
    print("Mode sequence:", modes)
    contact_faces_used = sorted(
        {m for m in modes if m.startswith("contact_face_")}
    )
    print("Distinct contact faces used:", contact_faces_used)

    if args.visualize:
        trajectory = history_to_trajectory(history, planner, initial_state, goal)
        visualize_planar_pushing_trajectory(
            trajectory, save=True, show=False, filename=args.output
        )
        print(f"Saved animation to {args.output}.mp4")


if __name__ == "__main__":
    main()
