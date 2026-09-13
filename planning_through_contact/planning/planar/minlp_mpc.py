"""Mixed-integer nonlinear MPC for a point pusher and rectangular slider.

This module deliberately lives beside, rather than inside, the existing GCS
planner.  It uses the same :class:`Box2d` face conventions and the same
ellipsoidal quasi-static limit-surface model as ``FaceContactMode``:

``v_WB = R_WB f_B / f_max**2`` and
``omega_WB = cross(p_Bc, f_B) / tau_max**2``.

The mode at each control interval is one of four face-contact modes or four
convex exterior free-space regions.  Big-M constraints activate the selected
mode, which lets BONMIN solve a compact direct MINLP.  The public warm-start
object is intentionally solver-independent so a learned mode/trajectory
predictor can replace its construction later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import casadi as ca
import numpy as np
import numpy.typing as npt

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.trajectory_builder import (
    OldPlanarPushingTrajectory,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    SliderPusherSystemConfig,
)


NUM_CONTACT_MODES = 4
NUM_FREE_MODES = 4
NUM_MODES = NUM_CONTACT_MODES + NUM_FREE_MODES


@dataclass(frozen=True)
class PlanarPushingState:
    """World-frame state ``[p_WB_x, p_WB_y, theta, p_WP_x, p_WP_y]``."""

    slider: PlanarPose
    pusher_position: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        pusher = np.asarray(self.pusher_position, dtype=float).reshape(-1)
        if pusher.shape != (2,):
            raise ValueError("pusher_position must contain exactly two coordinates")
        object.__setattr__(self, "pusher_position", pusher)

    @classmethod
    def from_vector(cls, value: npt.ArrayLike) -> "PlanarPushingState":
        x = np.asarray(value, dtype=float).reshape(-1)
        if x.shape != (5,):
            raise ValueError("A planar pushing state must have five entries")
        return cls(PlanarPose(x[0], x[1], x[2]), x[3:5])

    def vector(self) -> npt.NDArray[np.float64]:
        return np.array(
            [
                self.slider.x,
                self.slider.y,
                self.slider.theta,
                self.pusher_position[0],
                self.pusher_position[1],
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class PlanarPushingGoal:
    """Terminal slider pose used by the MPC objective."""

    slider: PlanarPose


@dataclass
class MinlpMpcConfig:
    """Numerical and cost settings for :class:`PlanarBoxPushingMinlpMpc`.

    ``max_normal_force`` is expressed in the repository's *scaled* contact
    force units.  The resulting physical bound is
    ``system.force_scale * max_normal_force``, matching ``FaceContactMode``.
    """

    horizon: int = 4
    time_step: float = 0.10
    position_lower: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([-0.5, -0.5])
    )
    position_upper: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.5, 0.5])
    )
    theta_lower: float = -np.pi
    theta_upper: float = np.pi
    max_pusher_speed: float = 0.5
    max_normal_force: float = 20.0
    big_m: float = 5.0
    terminal_position_weight: float = 2_000.0
    terminal_orientation_weight: float = 50.0
    running_position_weight: float = 2.0
    pusher_motion_weight: float = 1e-3
    force_weight: float = 1e-2
    mode_switch_weight: float = 0.1
    solver_time_limit: float = 5.0
    solver_feasibility_tolerance: float = 1e-5
    success_tolerance: float = 1e-3
    required_initial_mode: Optional[int] = None

    def __post_init__(self) -> None:
        self.position_lower = np.asarray(self.position_lower, dtype=float).reshape(-1)
        self.position_upper = np.asarray(self.position_upper, dtype=float).reshape(-1)
        if self.position_lower.shape != (2,) or self.position_upper.shape != (2,):
            raise ValueError("position_lower and position_upper must each have length 2")
        if np.any(self.position_lower >= self.position_upper):
            raise ValueError("position_lower must be strictly below position_upper")
        if self.horizon < 1 or self.time_step <= 0:
            raise ValueError("horizon and time_step must be positive")
        if self.max_normal_force <= 0 or self.big_m <= 0:
            raise ValueError("max_normal_force and big_m must be positive")
        if self.required_initial_mode is not None and not 0 <= self.required_initial_mode < NUM_MODES:
            raise ValueError(f"required_initial_mode must be in [0, {NUM_MODES})")


@dataclass
class MinlpMpcWarmStart:
    """Continuous and discrete initial guesses for one MPC solve.

    The arrays are directly consumable by this planner, but are also a useful
    contract for an eventual L2O predictor: predict a one-hot ``modes`` array
    plus the continuous trajectories, then pass it to :meth:`solve`.
    """

    states: npt.NDArray[np.float64]  # (5, horizon + 1)
    pusher_velocities: npt.NDArray[np.float64]  # (2, horizon)
    contact_forces_B: npt.NDArray[np.float64]  # (2, horizon), physical units
    contact_torques_B: npt.NDArray[np.float64]  # (horizon,)
    modes: npt.NDArray[np.float64]  # (8, horizon), one-hot columns

    def shifted(self) -> "MinlpMpcWarmStart":
        """Shift an MPC solution forward by one interval, repeating its tail."""

        def shift_columns(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.hstack((values[:, 1:], values[:, -1:]))

        return MinlpMpcWarmStart(
            states=shift_columns(self.states),
            pusher_velocities=shift_columns(self.pusher_velocities),
            contact_forces_B=shift_columns(self.contact_forces_B),
            contact_torques_B=np.hstack(
                (self.contact_torques_B[1:], self.contact_torques_B[-1:])
            ),
            modes=shift_columns(self.modes),
        )

    def with_initial_state(
        self, measured_state: PlanarPushingState
    ) -> "MinlpMpcWarmStart":
        """Replace the shifted initial knot with the newest measurement."""

        states = self.states.copy()
        states[:, 0] = measured_state.vector()
        return MinlpMpcWarmStart(
            states,
            self.pusher_velocities.copy(),
            self.contact_forces_B.copy(),
            self.contact_torques_B.copy(),
            self.modes.copy(),
        )


@dataclass
class MinlpMpcSolution:
    """A direct-MINLP prediction and its first executable input."""

    states: npt.NDArray[np.float64]
    pusher_velocities: npt.NDArray[np.float64]
    contact_forces_B: npt.NDArray[np.float64]
    contact_torques_B: npt.NDArray[np.float64]
    modes: npt.NDArray[np.float64]
    mode_indices: npt.NDArray[np.int_]
    objective: float
    solver_stats: dict

    @property
    def first_state(self) -> PlanarPushingState:
        return PlanarPushingState.from_vector(self.states[:, 0])

    @property
    def predicted_next_state(self) -> PlanarPushingState:
        return PlanarPushingState.from_vector(self.states[:, 1])

    @property
    def first_mode(self) -> int:
        return int(self.mode_indices[0])

    @property
    def warm_start(self) -> MinlpMpcWarmStart:
        return MinlpMpcWarmStart(
            self.states.copy(),
            self.pusher_velocities.copy(),
            self.contact_forces_B.copy(),
            self.contact_torques_B.copy(),
            self.modes.copy(),
        )


class MinlpSolveError(RuntimeError):
    """Raised if BONMIN does not return a successful feasible solution."""


class PlanarBoxPushingMinlpMpc:
    """Receding-horizon direct MINLP for the repository's rectangular slider.

    The planner uses one mode for every control interval.  Modes ``0..3`` are
    face contacts in the ordering supplied by :class:`Box2d`; modes ``4..7``
    are the corresponding collision-free regions.  A contact interval keeps
    the pusher fixed in the slider frame (sticking).  A free interval holds the
    slider and force fixed at zero and keeps both ends of the pusher segment in
    one convex exterior region, so the whole straight segment is collision-free.
    """

    def __init__(
        self,
        system: SliderPusherSystemConfig,
        config: Optional[MinlpMpcConfig] = None,
    ) -> None:
        if not isinstance(system.slider.geometry, Box2d):
            raise TypeError("PlanarBoxPushingMinlpMpc currently supports Box2d only")
        self.system = system
        self.config = config if config is not None else MinlpMpcConfig()
        self.geometry = system.slider.geometry
        self.mode_names = tuple(
            [f"contact_face_{i}" for i in range(NUM_CONTACT_MODES)]
            + [f"free_space_{i}" for i in range(NUM_FREE_MODES)]
        )

    @staticmethod
    def rotation(theta: ca.MX) -> ca.MX:
        return ca.vertcat(
            ca.horzcat(ca.cos(theta), -ca.sin(theta)),
            ca.horzcat(ca.sin(theta), ca.cos(theta)),
        )

    def mode_name(self, mode: int) -> str:
        return self.mode_names[mode]

    def mode_from_name(self, mode: str) -> int:
        try:
            return self.mode_names.index(mode)
        except ValueError as exc:
            raise ValueError(f"Unknown planar pushing mode: {mode}") from exc

    def solve(
        self,
        current_state: PlanarPushingState,
        goal: PlanarPushingGoal,
        *,
        previous_mode: Optional[int | str] = None,
        warm_start: Optional[MinlpMpcWarmStart] = None,
    ) -> MinlpMpcSolution:
        """Solve one short-horizon MINLP from the measured current state."""

        previous_mode_idx = self._normalise_mode(previous_mode)
        nlp, layout, bounds, discrete = self._build_problem(
            current_state, goal, previous_mode_idx
        )
        if warm_start is None:
            warm_start = self._make_default_warm_start(current_state, goal)
        else:
            warm_start = warm_start.with_initial_state(current_state)
        x0 = self._pack_warm_start(warm_start, layout, bounds[0], bounds[1])

        options = {
            "discrete": discrete,
            "print_time": False,
            # CasADi 3.8 exposes BONMIN options as one nested dictionary.  We
            # intentionally keep NLP-solver tuning out of this first version:
            # the outer MINLP time limit is the useful MPC safety bound.
            "bonmin": {
                "algorithm": "B-BB",
                "print_level": 0,
                "time_limit": self.config.solver_time_limit,
            },
        }
        solver = ca.nlpsol("planar_box_pushing_minlp", "bonmin", nlp, options)
        result = solver(
            x0=x0,
            lbx=bounds[0],
            ubx=bounds[1],
            lbg=bounds[2],
            ubg=bounds[3],
        )
        stats = solver.stats()
        feasible_incumbent = self._is_feasible_minlp_result(
            result, nlp, bounds, discrete
        )
        if not bool(stats.get("success", False)) and not feasible_incumbent:
            raise MinlpSolveError(
                f"BONMIN failed: {stats.get('return_status', 'unknown status')}"
            )
        if not bool(stats.get("success", False)):
            # In MPC a feasible time-limited incumbent is more useful than
            # discarding the solve.  Retain the status so a caller can decide
            # whether to accept it operationally.
            stats = {**stats, "accepted_feasible_incumbent": True}
        return self._unpack_solution(result, layout, stats)

    def run(
        self,
        initial_state: PlanarPushingState,
        goal: PlanarPushingGoal,
        *,
        max_iterations: int = 25,
        measurement_callback: Optional[
            Callable[[PlanarPushingState, MinlpMpcSolution], PlanarPushingState]
        ] = None,
    ) -> list[MinlpMpcSolution]:
        """Execute first controls repeatedly, using a measurement after each solve.

        With no callback this runs a model-consistent example: the next measured
        state is the predicted state after the first control.  A hardware or
        simulator integration should supply ``measurement_callback`` and return
        the newly measured state instead.
        """

        state = initial_state
        warm_start: Optional[MinlpMpcWarmStart] = None
        previous_mode: Optional[int] = None
        history: list[MinlpMpcSolution] = []
        for _ in range(max_iterations):
            if self.goal_reached(state, goal):
                break
            solution = self.solve(
                state,
                goal,
                previous_mode=previous_mode,
                warm_start=warm_start,
            )
            history.append(solution)
            executed_mode = solution.first_mode
            state = (
                measurement_callback(state, solution)
                if measurement_callback is not None
                else solution.predicted_next_state
            )
            warm_start = solution.warm_start.shifted()
            previous_mode = executed_mode
        return history

    def goal_reached(
        self, state: PlanarPushingState, goal: PlanarPushingGoal
    ) -> bool:
        pos_error = np.linalg.norm(state.slider.pos() - goal.slider.pos())
        angle_error = np.arctan2(
            np.sin(state.slider.theta - goal.slider.theta),
            np.cos(state.slider.theta - goal.slider.theta),
        )
        return bool(
            pos_error <= self.config.success_tolerance
            and abs(angle_error) <= self.config.success_tolerance
        )

    def to_legacy_trajectory(
        self, solutions: Sequence[MinlpMpcSolution]
    ) -> OldPlanarPushingTrajectory:
        """Adapt executed first steps for the repository's legacy visualizer."""

        if not solutions:
            raise ValueError("At least one MPC solution is required for visualization")
        states = np.column_stack(
            [solutions[0].states[:, 0]] + [solution.states[:, 1] for solution in solutions]
        )
        rotations = [
            np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            for theta in states[2]
        ]
        forces_W = np.column_stack(
            [
                rotations[k] @ solution.contact_forces_B[:, 0]
                for k, solution in enumerate(solutions)
            ]
            + [np.zeros(2)]
        )
        pusher_body = np.column_stack(
            [
                rotations[k].T @ (states[3:5, k] - states[0:2, k])
                for k in range(states.shape[1])
            ]
        )
        return OldPlanarPushingTrajectory(
            self.config.time_step,
            rotations,
            states[0:2],
            states[3:5],
            forces_W,
            pusher_body,
        )

    def solution_to_legacy_trajectory(
        self, solution: MinlpMpcSolution
    ) -> OldPlanarPushingTrajectory:
        """Adapt one complete open-loop MINLP solution for legacy visualization.

        Unlike :meth:`to_legacy_trajectory`, which collects the first executed
        interval from each MPC solve, this preserves every interval of a single
        long-horizon solution.
        """

        h = self.config.horizon
        expected_shapes = {
            "states": (5, h + 1),
            "contact_forces_B": (2, h),
        }
        invalid_shapes = {
            name: np.asarray(getattr(solution, name)).shape
            for name, expected in expected_shapes.items()
            if np.asarray(getattr(solution, name)).shape != expected
        }
        if invalid_shapes:
            raise ValueError(
                "Solution has shapes incompatible with this planner horizon: "
                f"{invalid_shapes}; expected {expected_shapes}"
            )

        states = solution.states
        rotations = [
            np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            for theta in states[2]
        ]
        forces_W = np.column_stack(
            [
                rotations[k] @ solution.contact_forces_B[:, k]
                for k in range(h)
            ]
            + [np.zeros(2)]
        )
        pusher_body = np.column_stack(
            [
                rotations[k].T @ (states[3:5, k] - states[0:2, k])
                for k in range(h + 1)
            ]
        )
        return OldPlanarPushingTrajectory(
            self.config.time_step,
            rotations,
            states[0:2],
            states[3:5],
            forces_W,
            pusher_body,
        )

    def _build_problem(
        self,
        current: PlanarPushingState,
        goal: PlanarPushingGoal,
        previous_mode: Optional[int],
    ) -> tuple[
        dict,
        dict[str, tuple[int, tuple[int, ...]]],
        tuple[np.ndarray, ...],
        list[bool],
    ]:
        h = self.config.horizon
        X = ca.MX.sym("X", 5, h + 1)
        U = ca.MX.sym("U", 2, h)
        F = ca.MX.sym("F_B", 2, h)
        TAU = ca.MX.sym("tau_B", h)
        Z = ca.MX.sym("mode", NUM_MODES, h)
        SWITCH = ca.MX.sym("switch", max(h - 1, 0))

        variables = [X, U, F, TAU, Z, SWITCH]
        x = ca.vertcat(*[ca.vec(v) for v in variables])
        layout: dict[str, tuple[int, tuple[int, ...]]] = {}
        offset = 0
        for name, value in zip(("X", "U", "F", "TAU", "Z", "SWITCH"), variables):
            layout[name] = (offset, value.shape)
            offset += int(value.numel())

        g: list[ca.MX] = []
        lbg: list[float] = []
        ubg: list[float] = []

        def add_constraint(
            expr: ca.MX, lb: float | npt.ArrayLike, ub: float | npt.ArrayLike
        ) -> None:
            flat = ca.vec(expr)
            size = int(flat.numel())
            lower = np.broadcast_to(np.asarray(lb, dtype=float), (size,)).tolist()
            upper = np.broadcast_to(np.asarray(ub, dtype=float), (size,)).tolist()
            g.append(flat)
            lbg.extend(lower)
            ubg.extend(upper)

        def add_equality(expr: ca.MX) -> None:
            add_constraint(expr, 0.0, 0.0)

        cfg = self.config
        system = self.system
        dt = cfg.time_step
        force_max = system.force_scale * cfg.max_normal_force
        torque_max = force_max * (system.max_contact_radius + system.pusher_radius)
        m = cfg.big_m

        # Initial condition and original quasi-static ellipsoidal limit surface.
        add_equality(X[:, 0] - current.vector())
        for k in range(h):
            R = self.rotation(X[2, k])
            add_equality(
                X[0:2, k + 1]
                - X[0:2, k]
                - dt * R @ (F[:, k] / system.f_max**2)
            )
            add_equality(
                X[2, k + 1]
                - X[2, k]
                - dt * TAU[k] / system.tau_max**2
            )
            add_equality(X[3:5, k + 1] - X[3:5, k] - dt * U[:, k])
            add_constraint(ca.sumsqr(U[:, k]), 0.0, cfg.max_pusher_speed**2)
            add_equality(ca.sum1(Z[:, k]) - 1)

            contact_active = ca.sum1(Z[:NUM_CONTACT_MODES, k])
            # A free-space mode has zero input force/torque, and therefore the
            # quasi-static dynamics above hold the slider stationary.
            add_constraint(F[:, k] - force_max * contact_active, -np.inf, 0.0)
            add_constraint(F[:, k] + force_max * contact_active, 0.0, np.inf)
            add_constraint(TAU[k] - torque_max * contact_active, -np.inf, 0.0)
            add_constraint(TAU[k] + torque_max * contact_active, 0.0, np.inf)

            p_BP = R.T @ (X[3:5, k] - X[0:2, k])
            R_next = self.rotation(X[2, k + 1])
            p_BP_next = R_next.T @ (X[3:5, k + 1] - X[0:2, k + 1])

            for face_idx in range(NUM_CONTACT_MODES):
                z_contact = Z[face_idx, k]
                location = PolytopeContactLocation(ContactLocation.FACE, face_idx)
                v1, v2 = self.geometry.get_proximate_vertices_from_location(location)
                normal, tangent = self.geometry.get_norm_and_tang_vecs_from_location(
                    location
                )
                normal = ca.DM(normal.reshape(2, 1))
                tangent = ca.DM(tangent.reshape(2, 1))
                p_Bc = p_BP + normal * system.pusher_radius

                # Selected contact: a finite face at one pusher radius outside
                # the box, Coulomb cone, torque, and sticking in the body frame.
                face_plane = ca.dot(normal, p_Bc - ca.DM(v1.reshape(2, 1)))
                add_constraint(face_plane + m * (1 - z_contact), 0.0, np.inf)
                add_constraint(-face_plane + m * (1 - z_contact), 0.0, np.inf)
                tangent_coordinate = ca.dot(tangent, p_Bc)
                t1 = float(tangent.T @ v1)
                t2 = float(tangent.T @ v2)
                tangent_lb, tangent_ub = min(t1, t2), max(t1, t2)
                add_constraint(
                    tangent_coordinate - tangent_lb + m * (1 - z_contact),
                    0.0,
                    np.inf,
                )
                add_constraint(
                    tangent_ub - tangent_coordinate + m * (1 - z_contact),
                    0.0,
                    np.inf,
                )
                normal_force = ca.dot(normal, F[:, k])
                friction_force = ca.dot(tangent, F[:, k])
                add_constraint(normal_force + m * (1 - z_contact), 0.0, np.inf)
                add_constraint(
                    force_max + m * (1 - z_contact) - normal_force, 0.0, np.inf
                )
                add_constraint(
                    m * (1 - z_contact)
                    - friction_force
                    + system.friction_coeff_slider_pusher * normal_force,
                    0.0,
                    np.inf,
                )
                add_constraint(
                    m * (1 - z_contact)
                    + friction_force
                    + system.friction_coeff_slider_pusher * normal_force,
                    0.0,
                    np.inf,
                )
                torque = p_Bc[0] * F[1, k] - p_Bc[1] * F[0, k]
                add_constraint(
                    TAU[k] - torque + m * (1 - z_contact), 0.0, np.inf
                )
                add_constraint(
                    torque - TAU[k] + m * (1 - z_contact), 0.0, np.inf
                )
                add_constraint(
                    p_BP_next - p_BP + m * (1 - z_contact), 0.0, np.inf
                )
                add_constraint(
                    p_BP - p_BP_next + m * (1 - z_contact), 0.0, np.inf
                )

            for region_idx in range(NUM_FREE_MODES):
                z_free = Z[NUM_CONTACT_MODES + region_idx, k]
                # This is the same exterior wedge decomposition used by
                # NonCollisionMode.  Applying it at both endpoints makes the
                # straight pusher segment collision-free while the slider rests.
                planes = [self.geometry.faces[region_idx]] + list(
                    self.geometry.get_planes_for_collision_free_region(region_idx)
                )
                for plane_idx, plane in enumerate(planes):
                    clearance = system.pusher_radius if plane_idx == 0 else 0.0
                    a = ca.DM(plane.a.reshape(2, 1))
                    b = float(plane.b.item())
                    for pusher_body_position in (p_BP, p_BP_next):
                        add_constraint(
                            ca.dot(a, pusher_body_position)
                            - b
                            - clearance
                            + m * (1 - z_free),
                            0.0,
                            np.inf,
                        )

        # Contact can persist or be released into free space, but a direct
        # switch to a different face is prohibited; regrasping must use a free
        # interval.  Free-space region changes are only feasible at an actual
        # geometric intersection because both adjacent stages constrain the
        # shared pusher state.
        for k in range(h - 1):
            for first_face in range(NUM_CONTACT_MODES):
                for second_face in range(NUM_CONTACT_MODES):
                    if first_face != second_face:
                        add_constraint(
                            Z[first_face, k] + Z[second_face, k + 1], -np.inf, 1.0
                        )
            for mode_idx in range(NUM_MODES):
                add_constraint(
                    SWITCH[k] - Z[mode_idx, k] + Z[mode_idx, k + 1], 0.0, np.inf
                )

        if previous_mode is not None:
            if previous_mode < NUM_CONTACT_MODES:
                for face_idx in range(NUM_CONTACT_MODES):
                    if face_idx != previous_mode:
                        add_constraint(Z[face_idx, 0], 0.0, 0.0)
            else:
                previous_region = previous_mode - NUM_CONTACT_MODES
                for face_idx in range(NUM_CONTACT_MODES):
                    if face_idx != previous_region:
                        add_constraint(Z[face_idx, 0], 0.0, 0.0)
        if cfg.required_initial_mode is not None:
            add_constraint(Z[cfg.required_initial_mode, 0], 1.0, 1.0)

        goal_xy = ca.DM(goal.slider.pos().reshape(2, 1))
        terminal_position_error = X[0:2, -1] - goal_xy
        terminal_angle_error = X[2, -1] - goal.slider.theta
        cost = (
            cfg.terminal_position_weight * ca.sumsqr(terminal_position_error)
            + cfg.terminal_orientation_weight * (1 - ca.cos(terminal_angle_error))
            + cfg.pusher_motion_weight * ca.sumsqr(U)
            + cfg.force_weight * ca.sumsqr(F)
        )
        for k in range(h):
            cost += cfg.running_position_weight * ca.sumsqr(X[0:2, k] - goal_xy)
        if h > 1:
            cost += cfg.mode_switch_weight * ca.sum1(SWITCH)
        if previous_mode is not None:
            cost += cfg.mode_switch_weight * (1 - Z[previous_mode, 0])

        lbx, ubx, discrete = self._decision_bounds(variables, force_max, torque_max)
        return (
            {"x": x, "f": cost, "g": ca.vertcat(*g)},
            layout,
            (lbx, ubx, np.asarray(lbg), np.asarray(ubg)),
            discrete,
        )

    def _decision_bounds(
        self, variables: Sequence[ca.MX], force_max: float, torque_max: float
    ) -> tuple[np.ndarray, np.ndarray, list[bool]]:
        cfg = self.config
        h = cfg.horizon
        state_lb = np.array(
            [*cfg.position_lower, cfg.theta_lower, *cfg.position_lower], dtype=float
        )
        state_ub = np.array(
            [*cfg.position_upper, cfg.theta_upper, *cfg.position_upper], dtype=float
        )
        bounds = [
            (np.tile(state_lb, h + 1), np.tile(state_ub, h + 1), False),
            (
                np.tile(np.full(2, -cfg.max_pusher_speed), h),
                np.tile(np.full(2, cfg.max_pusher_speed), h),
                False,
            ),
            (
                np.tile(np.full(2, -force_max), h),
                np.tile(np.full(2, force_max), h),
                False,
            ),
            (np.full(h, -torque_max), np.full(h, torque_max), False),
            (np.zeros(NUM_MODES * h), np.ones(NUM_MODES * h), True),
            (np.zeros(max(h - 1, 0)), np.ones(max(h - 1, 0)), False),
        ]
        assert len(bounds) == len(variables)
        lbx = np.concatenate([item[0] for item in bounds])
        ubx = np.concatenate([item[1] for item in bounds])
        discrete = [item[2] for item in bounds for _ in range(len(item[0]))]
        return lbx, ubx, discrete

    def _make_default_warm_start(
        self, current: PlanarPushingState, goal: PlanarPushingGoal
    ) -> MinlpMpcWarmStart:
        h = self.config.horizon
        alpha = np.linspace(0.0, 1.0, h + 1)
        current_vector = current.vector()
        states = np.empty((5, h + 1))
        states[0] = current.slider.x + alpha * (goal.slider.x - current.slider.x)
        states[1] = current.slider.y + alpha * (goal.slider.y - current.slider.y)
        states[2] = current.slider.theta + alpha * (
            goal.slider.theta - current.slider.theta
        )
        R0 = np.array(
            [
                [np.cos(current.slider.theta), -np.sin(current.slider.theta)],
                [np.sin(current.slider.theta), np.cos(current.slider.theta)],
            ]
        )
        p_BP_initial = R0.T @ (
            current.pusher_position - current.slider.pos().flatten()
        )
        for k in range(h + 1):
            R = np.array(
                [
                    [np.cos(states[2, k]), -np.sin(states[2, k])],
                    [np.sin(states[2, k]), np.cos(states[2, k])],
                ]
            )
            states[3:5, k] = states[0:2, k] + R @ p_BP_initial
        states[:, 0] = current_vector
        velocities = np.diff(states[3:5], axis=1) / self.config.time_step
        forces = np.zeros((2, h))
        for k in range(h):
            R = np.array(
                [
                    [np.cos(states[2, k]), -np.sin(states[2, k])],
                    [np.sin(states[2, k]), np.cos(states[2, k])],
                ]
            )
            slider_velocity = (
                states[0:2, k + 1] - states[0:2, k]
            ) / self.config.time_step
            forces[:, k] = self.system.f_max**2 * R.T @ slider_velocity
        torques = np.diff(states[2]) * self.system.tau_max**2 / self.config.time_step
        mode = self._closest_mode(p_BP_initial)
        modes = np.zeros((NUM_MODES, h))
        modes[mode, :] = 1.0
        return MinlpMpcWarmStart(states, velocities, forces, torques, modes)

    def _closest_mode(self, p_BP: npt.NDArray[np.float64]) -> int:
        distances = []
        for face_idx in range(NUM_CONTACT_MODES):
            loc = PolytopeContactLocation(ContactLocation.FACE, face_idx)
            v1, _ = self.geometry.get_proximate_vertices_from_location(loc)
            normal, _ = self.geometry.get_norm_and_tang_vecs_from_location(loc)
            p_expected_plane = v1 - normal * self.system.pusher_radius
            signed_distance = normal.T @ (
                p_BP.reshape(2, 1) - p_expected_plane
            )
            distances.append(abs(float(signed_distance.item())))
        return int(np.argmin(distances))

    def _pack_warm_start(
        self,
        warm: MinlpMpcWarmStart,
        layout: dict[str, tuple[int, tuple[int, ...]]],
        lbx: np.ndarray,
        ubx: np.ndarray,
    ) -> np.ndarray:
        h = self.config.horizon
        expected_shapes = {
            "X": (5, h + 1),
            "U": (2, h),
            "F": (2, h),
            "TAU": (h,),
            "Z": (NUM_MODES, h),
        }
        values = {
            "X": warm.states,
            "U": warm.pusher_velocities,
            "F": warm.contact_forces_B,
            "TAU": warm.contact_torques_B,
            "Z": warm.modes,
            "SWITCH": np.zeros(max(h - 1, 0)),
        }
        packed = np.zeros(sum(int(np.prod(shape)) for _, shape in layout.values()))
        for name, (start, shape) in layout.items():
            value = np.asarray(values[name], dtype=float)
            if name in expected_shapes and value.shape != expected_shapes[name]:
                raise ValueError(
                    f"Warm-start {name} has shape {value.shape}; expected {expected_shapes[name]}"
                )
            packed[start : start + int(np.prod(shape))] = value.reshape(-1, order="F")
        return np.clip(packed, lbx, ubx)

    def _unpack_solution(
        self, result: dict, layout: dict[str, tuple[int, tuple[int, ...]]], stats: dict
    ) -> MinlpMpcSolution:
        x = np.asarray(result["x"].full(), dtype=float).reshape(-1)

        def value(name: str) -> np.ndarray:
            start, shape = layout[name]
            size = int(np.prod(shape))
            return x[start : start + size].reshape(shape, order="F")

        modes = np.rint(value("Z"))
        return MinlpMpcSolution(
            states=value("X"),
            pusher_velocities=value("U"),
            contact_forces_B=value("F"),
            contact_torques_B=value("TAU").reshape(-1),
            modes=modes,
            mode_indices=np.argmax(modes, axis=0).astype(int),
            objective=float(result["f"]),
            solver_stats=stats,
        )

    def _is_feasible_minlp_result(
        self,
        result: dict,
        nlp: dict,
        bounds: tuple[np.ndarray, ...],
        discrete: Sequence[bool],
    ) -> bool:
        """Check a feasible incumbent returned despite a BONMIN time limit."""

        x = np.asarray(result["x"].full(), dtype=float).reshape(-1)
        # CasADi leaves ``result["g"]`` as NaN after BONMIN returns a
        # time-limited incumbent, so evaluate the original constraints at the
        # returned decision vector instead.
        constraint_evaluator = ca.Function(
            "minlp_constraint_evaluator", [nlp["x"]], [nlp["g"]]
        )
        g = np.asarray(
            constraint_evaluator(result["x"]).full(), dtype=float
        ).reshape(-1)
        _, _, lbg, ubg = bounds
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(g)):
            return False
        constraint_violation = max(
            float(np.max(np.maximum(lbg - g, 0.0))),
            float(np.max(np.maximum(g - ubg, 0.0))),
        )
        discrete_values = x[np.asarray(discrete, dtype=bool)]
        integrality_violation = float(
            np.max(np.abs(discrete_values - np.rint(discrete_values)))
        )
        return bool(
            constraint_violation <= self.config.solver_feasibility_tolerance
            and integrality_violation <= self.config.solver_feasibility_tolerance
        )

    def _normalise_mode(self, mode: Optional[int | str]) -> Optional[int]:
        if mode is None:
            return None
        idx = self.mode_from_name(mode) if isinstance(mode, str) else int(mode)
        if not 0 <= idx < NUM_MODES:
            raise ValueError(f"Mode index must be in [0, {NUM_MODES}), got {idx}")
        return idx
