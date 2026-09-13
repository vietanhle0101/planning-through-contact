"""Renders the converged open-loop forced-regrasp plan with the modern
(SceneGraph-based) visualizer, since run_minlp_open_loop_box.py's --visualize
uses the legacy Tkinter visualizer, which requires an interactive display and
never returns (no headless/mp4 output) -- confirmed by hanging indefinitely
when tried earlier in this session.

Re-solves the exact scenario from that script's forced-regrasp run:
    --solver gurobi --horizon 18 --time-limit 300
    --goal-x -0.12 --goal-y 0.0 --goal-theta-deg 0
    (start: box at origin, pusher in contact with face 3)
which converged to Goal reached: True with mode schedule
free_space_3 -> free_space_0 -> free_space_1 -> contact_face_1.
"""

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


def main() -> None:
    box = Box2d(width=0.20, height=0.10)
    system = SliderPusherSystemConfig(
        slider=RigidBody("minlp_box", box, mass=0.10),
        pusher_radius=0.015,
        friction_coeff_slider_pusher=0.30,
        friction_coeff_table_slider=0.50,
        integration_constant=0.60,
    )
    config = MinlpMpcConfig(
        horizon=18,
        time_step=0.10,
        position_lower=np.array([-0.40, -0.40]),
        position_upper=np.array([0.40, 0.40]),
        solver_time_limit=300.0,
        success_tolerance=0.015,
    )
    planner = PlanarBoxPushingGurobiMpc(system, config)

    initial_slider = PlanarPose(0.0, 0.0, 0.0)
    initial_state = PlanarPushingState(
        slider=initial_slider,
        pusher_position=initial_slider.pos().flatten() + np.array([-0.115, 0.0]),
    )
    goal = PlanarPushingGoal(PlanarPose(-0.12, 0.0, 0.0))

    print("Re-solving (this took ~5 minutes previously with a 300s Gurobi budget)...")
    solution = planner.solve(initial_state, goal)

    terminal = PlanarPushingState.from_vector(solution.states[:, -1])
    print(f"Terminal pose: {terminal.slider}")
    print(f"Goal reached: {planner.goal_reached(terminal, goal)}")
    print("Mode sequence:", [planner.mode_name(m) for m in solution.mode_indices])

    # Build a trajectory directly from the whole open-loop solution (all 19
    # knots), unlike the MPC scripts which stitch together only the executed
    # first step of each receding-horizon re-solve.
    states = solution.states  # (5, horizon+1)
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

    output = "trajectories/minlp_open_loop_forced_regrasp"
    visualize_planar_pushing_trajectory(trajectory, save=True, show=False, filename=output)
    print(f"Saved animation to {output}.mp4")


if __name__ == "__main__":
    main()
