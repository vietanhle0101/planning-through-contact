import numpy as np

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.minlp_mpc import (
    MinlpMpcConfig,
    MinlpMpcWarmStart,
    PlanarBoxPushingMinlpMpc,
    PlanarPushingGoal,
    PlanarPushingState,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    SliderPusherSystemConfig,
)


def _make_problem(horizon: int = 2) -> tuple[
    PlanarBoxPushingMinlpMpc, PlanarPushingState, PlanarPushingGoal
]:
    system = SliderPusherSystemConfig(
        slider=RigidBody("box", Box2d(0.20, 0.10), mass=0.10),
        pusher_radius=0.015,
        friction_coeff_slider_pusher=0.30,
        friction_coeff_table_slider=0.50,
        integration_constant=0.60,
    )
    planner = PlanarBoxPushingMinlpMpc(
        system,
        MinlpMpcConfig(
            horizon=horizon,
            time_step=0.10,
            position_lower=np.array([-0.4, -0.4]),
            position_upper=np.array([0.4, 0.4]),
            solver_time_limit=5.0,
            success_tolerance=2e-3,
        ),
    )
    initial_state = PlanarPushingState(
        PlanarPose(0.0, 0.0, 0.0), np.array([-0.115, 0.0])
    )
    goal = PlanarPushingGoal(PlanarPose(0.04, 0.0, 0.0))
    return planner, initial_state, goal


def test_minlp_selects_the_feasible_pushing_face_and_respects_sticking() -> None:
    planner, initial_state, goal = _make_problem()
    solution = planner.solve(initial_state, goal)

    assert np.all(solution.mode_indices == 3)  # left box face
    assert solution.states[0, -1] > 0.039
    assert np.all(solution.contact_forces_B[0] >= -1e-7)

    # A sticking contact leaves the pusher center at the same body-frame point.
    relative_positions = []
    for state in solution.states.T:
        rotation = np.array(
            [
                [np.cos(state[2]), -np.sin(state[2])],
                [np.sin(state[2]), np.cos(state[2])],
            ]
        )
        relative_positions.append(rotation.T @ (state[3:5] - state[:2]))
    assert np.allclose(relative_positions, np.array([-0.115, 0.0]), atol=2e-4)

    trajectory = planner.solution_to_legacy_trajectory(solution)
    assert trajectory.N == planner.config.horizon + 1
    assert trajectory.f_c_W.shape == (2, planner.config.horizon + 1)


def test_mpc_carries_executed_mode_and_shifts_the_solution() -> None:
    planner, initial_state, goal = _make_problem()
    history = planner.run(initial_state, goal, max_iterations=4)

    assert len(history) == 2
    assert history[0].first_mode == 3
    assert history[1].first_mode == 3
    assert np.allclose(history[1].states[:, 0], history[0].states[:, 1])
    assert planner.goal_reached(history[-1].predicted_next_state, goal)


def test_minlp_can_regrasp_through_a_free_space_mode() -> None:
    planner, _, goal = _make_problem(horizon=3)
    initial_state = PlanarPushingState(
        PlanarPose(0.0, 0.0, 0.0), np.array([-0.150, 0.0])
    )
    # A mode-and-continuous warm start is the contract an L2O initializer will
    # eventually provide: move through free region 3, then contact face 3.
    force_for_2cm_step = planner.system.f_max**2 * 0.20
    warm_start = MinlpMpcWarmStart(
        states=np.array(
            [
                [0.0, 0.0, 0.02, 0.04],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-0.150, -0.115, -0.095, -0.075],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        pusher_velocities=np.array([[0.35, 0.20, 0.20], [0.0, 0.0, 0.0]]),
        contact_forces_B=np.array(
            [[0.0, force_for_2cm_step, force_for_2cm_step], [0.0, 0.0, 0.0]]
        ),
        contact_torques_B=np.zeros(3),
        modes=np.eye(8)[:, [7, 3, 3]],
    )

    solution = planner.solve(initial_state, goal, warm_start=warm_start)

    assert solution.mode_indices.tolist() == [7, 3, 3]
    assert np.isclose(solution.states[0, 1], 0.0, atol=1e-5)
    assert solution.states[0, -1] > 0.039
