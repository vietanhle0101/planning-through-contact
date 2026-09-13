"""A scenario that structurally requires a regrasp, not just prefers one.

run_minlp_mpc_box_challenging.py's target could apparently be reached (to
within tolerance) by pushing on a single face the whole way -- whether the
MPC finds the two-face solution or the single-face one turned out to depend
on horizon length and even on incidental solver-configuration details (see
the discussion around minlp_mpc.py's BONMIN options).

This scenario removes that ambiguity. The pusher starts in contact with face
3 (the box's left face). A face's contact-force normal component is
constrained to be non-negative (c_n >= 0: a pusher can only push, never
pull), and face 3's inward normal points in +x. So *no* sequence of forces
applied through face 3 can ever move the box to a smaller x than it already
has -- moving further left is a hard infeasibility, not a friction-cone
inefficiency. The goal below requires the box to move to x = -0.12 from a
start at x = 0.0, which is only reachable by releasing face 3, walking the
pusher around the box in free space, and pushing from a face whose normal
has a -x component (face 1, the right face, or one of its free-space
neighbors' induced paths).

This makes "does the mode sequence actually switch faces" a hard yes/no
question rather than a "which of two similarly-costed plans did the solver
happen to return" question.
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


def rotation_matrix(theta: float) -> np.ndarray:
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


def make_planner(horizon: int = 3) -> PlanarBoxPushingMinlpMpc:
    box = Box2d(width=0.20, height=0.10)
    system = SliderPusherSystemConfig(
        slider=RigidBody("minlp_box", box, mass=0.10),
        pusher_radius=0.015,
        friction_coeff_slider_pusher=0.30,
        friction_coeff_table_slider=0.50,
        integration_constant=0.60,
    )
    mpc_config = MinlpMpcConfig(
        horizon=horizon,
        time_step=0.10,
        position_lower=np.array([-0.50, -0.50]),
        position_upper=np.array([0.50, 0.50]),
        max_pusher_speed=0.6,
        solver_time_limit=8.0,
        success_tolerance=1.5e-2,
    )
    return PlanarBoxPushingMinlpMpc(system, mpc_config)


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
    R_WBs = [rotation_matrix(theta) for theta in states[2]]
    p_WPs = states[3:5].T
    f_c_Ws = np.column_stack(
        [R_WBs[k] @ sol.contact_forces_B[:, 0] for k, sol in enumerate(history)]
        + [np.zeros(2)]
    ).T
    plan_config = PlanarPlanConfig(
        dynamics_config=planner.system,
        start_and_goal=PlanarPushingStartAndGoal(
            slider_initial_pose=initial_state.slider,
            slider_target_pose=goal.slider,
            pusher_initial_pose=PlanarPose(*initial_state.pusher_position, 0.0),
            pusher_target_pose=PlanarPose(
                *history[-1].predicted_next_state.pusher_position, 0.0
            ),
        ),
    )
    return SimplePlanarPushingTrajectory(
        p_WBs, R_WBs, p_WPs, f_c_Ws, planner.config.time_step, plan_config
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--output", type=str, default="trajectories/minlp_mpc_forced_regrasp"
    )
    args = parser.parse_args()

    planner = make_planner(args.horizon)
    initial_state = PlanarPushingState(
        slider=PlanarPose(0.0, 0.0, 0.0),
        pusher_position=np.array([-0.115, 0.0]),  # in contact with face 3
    )
    # Face 3's normal points in +x, so c_n >= 0 makes any x < 0.0 structurally
    # unreachable while in contact with face 3: this is a hard constraint,
    # not a cost trade-off.
    goal = PlanarPushingGoal(PlanarPose(-0.12, 0.0, 0.0))

    print(f"Start: {initial_state.slider} (in contact with face 3, normal = +x)")
    print(f"Goal:  {goal.slider}  (x < start x: infeasible from face 3 alone)")

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
    modes = [planner.mode_name(step.first_mode) for step in history]
    contact_faces_used = sorted({m for m in modes if m.startswith("contact_face_")})
    print(f"\nExecuted {len(history)} MPC iterations in {elapsed:.1f}s")
    print(f"Final slider pose: {final_state.slider}")
    print(
        f"Goal reached (tol={planner.config.success_tolerance}): "
        f"{planner.goal_reached(final_state, goal)}"
    )
    print("Mode sequence:", modes)
    print("Distinct contact faces used:", contact_faces_used)
    print(
        "Regrasp confirmed (more than one face used):",
        len(contact_faces_used) > 1,
    )

    if args.visualize:
        trajectory = history_to_trajectory(history, planner, initial_state, goal)
        visualize_planar_pushing_trajectory(
            trajectory, save=True, show=False, filename=args.output
        )
        print(f"Saved animation to {args.output}.mp4")


if __name__ == "__main__":
    main()
