"""Compares the BONMIN and Gurobi backends on the same challenging scenario.

Uses the exact box/friction/MPC-config parameters and start/goal pose from
run_minlp_mpc_box_challenging.py (translate 15cm/8cm and rotate 30 degrees,
which that script showed requires the BONMIN backend to regrasp from face 3
to face 2). Runs the identical receding-horizon loop with both
PlanarBoxPushingMinlpMpc (BONMIN, via CasADi) and PlanarBoxPushingGurobiMpc
(native Gurobi nonconvex MINLP) and reports wall-clock time, iteration count,
final pose error, and the executed mode sequence for each.

Does not modify minlp_mpc.py or gurobi_minlp_mpc.py.
"""

import time

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

# Identical to run_minlp_mpc_box_challenging.py.
INITIAL_STATE = PlanarPushingState(
    slider=PlanarPose(0.0, 0.0, 0.0),
    pusher_position=np.array([-0.115, 0.0]),
)
GOAL = PlanarPushingGoal(PlanarPose(0.15, 0.08, np.deg2rad(30)))
MAX_ITERATIONS = 20
SOLVER_TIME_LIMIT = 8.0


def make_system() -> SliderPusherSystemConfig:
    box = Box2d(width=0.20, height=0.10)
    return SliderPusherSystemConfig(
        slider=RigidBody("minlp_box", box, mass=0.10),
        pusher_radius=0.015,
        friction_coeff_slider_pusher=0.30,
        friction_coeff_table_slider=0.50,
        integration_constant=0.60,
    )


def make_config() -> MinlpMpcConfig:
    return MinlpMpcConfig(
        horizon=3,
        time_step=0.10,
        position_lower=np.array([-0.50, -0.50]),
        position_upper=np.array([0.50, 0.50]),
        max_pusher_speed=0.6,
        solver_time_limit=SOLVER_TIME_LIMIT,
        success_tolerance=1e-2,
    )


def run_backend(name: str, planner) -> dict:
    print(f"\n=== {name} ===")
    state = INITIAL_STATE
    warm_start = None
    previous_mode = None
    per_iter_times: list[float] = []
    modes: list[str] = []
    start_time = time.time()
    for iteration in range(MAX_ITERATIONS):
        if planner.goal_reached(state, GOAL):
            break
        iter_start = time.time()
        try:
            solution = planner.solve(
                state, GOAL, previous_mode=previous_mode, warm_start=warm_start
            )
        except MinlpSolveError as exc:
            print(f"  Iter {iteration}: solve failed ({exc}); stopping.")
            break
        iter_time = time.time() - iter_start
        per_iter_times.append(iter_time)
        mode_name = planner.mode_name(solution.first_mode)
        modes.append(mode_name)
        next_state = solution.predicted_next_state
        pos_error = np.linalg.norm(next_state.slider.pos() - GOAL.slider.pos())
        print(
            f"  Iter {iteration:2d} ({iter_time:5.2f}s): mode={mode_name:<16s} "
            f"pose=({next_state.slider.x:+.3f}, {next_state.slider.y:+.3f}, "
            f"{np.rad2deg(next_state.slider.theta):+.1f} deg)  "
            f"pos_err={pos_error * 100:.2f} cm"
        )
        state = next_state
        warm_start = solution.warm_start.shifted()
        previous_mode = solution.first_mode

    total_time = time.time() - start_time
    reached = planner.goal_reached(state, GOAL)
    return {
        "name": name,
        "num_iterations": len(modes),
        "total_time": total_time,
        "per_iter_times": per_iter_times,
        "final_state": state,
        "goal_reached": reached,
        "modes": modes,
    }


def main() -> None:
    results = []
    for name, planner_cls in [
        ("BONMIN (CasADi)", PlanarBoxPushingMinlpMpc),
        ("Gurobi", PlanarBoxPushingGurobiMpc),
    ]:
        planner = planner_cls(make_system(), make_config())
        results.append(run_backend(name, planner))

    print("\n" + "=" * 78)
    print(f"{'Backend':<18}{'Iters':>7}{'Total (s)':>12}{'Avg/iter (s)':>15}{'Reached':>10}")
    print("-" * 78)
    for r in results:
        avg = np.mean(r["per_iter_times"]) if r["per_iter_times"] else float("nan")
        print(
            f"{r['name']:<18}{r['num_iterations']:>7}{r['total_time']:>12.2f}"
            f"{avg:>15.2f}{str(r['goal_reached']):>10}"
        )
    print("=" * 78)
    for r in results:
        print(f"\n{r['name']} final pose: {r['final_state'].slider}")
        print(f"{r['name']} mode sequence: {r['modes']}")


if __name__ == "__main__":
    main()
