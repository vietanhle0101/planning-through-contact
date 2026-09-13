"""Native Gurobi backend for the rectangular-box direct MINLP MPC.

Gurobi cannot consume the CasADi MINLP directly, so this module transcribes the
same model with gurobipy variables and constraints.  It intentionally shares
the state, configuration, warm-start, result, and MPC-loop interfaces with the
BONMIN implementation in :mod:`minlp_mpc`.
"""

from __future__ import annotations

from typing import Optional

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.planning.planar.minlp_mpc import (
    NUM_CONTACT_MODES,
    NUM_FREE_MODES,
    NUM_MODES,
    MinlpMpcSolution,
    MinlpMpcWarmStart,
    MinlpSolveError,
    PlanarBoxPushingMinlpMpc,
    PlanarPushingGoal,
    PlanarPushingState,
)


class PlanarBoxPushingGurobiMpc(PlanarBoxPushingMinlpMpc):
    """The direct rectangular-box MINLP transcribed for native Gurobi.

    The formulation is equivalent to :class:`PlanarBoxPushingMinlpMpc`, except
    that explicit ``cos(theta)``, ``sin(theta)``, and body-frame pusher
    variables expose most relations as bilinear/quadratic constraints.  Gurobi
    handles the remaining sine/cosine definitions as nonlinear constraints.

    A Gurobi 13+ license is required at solve time.  The inherited ``run`` and
    ``to_legacy_trajectory`` methods provide the same MPC and visualization
    behavior as the BONMIN backend.
    """

    def solve(
        self,
        current_state: PlanarPushingState,
        goal: PlanarPushingGoal,
        *,
        previous_mode: Optional[int | str] = None,
        warm_start: Optional[MinlpMpcWarmStart] = None,
    ) -> MinlpMpcSolution:
        """Solve one MPC MINLP with Gurobi's nonconvex global machinery."""

        previous_mode_idx = self._normalise_mode(previous_mode)
        if warm_start is None:
            warm_start = self._make_default_warm_start(current_state, goal)
        else:
            warm_start = warm_start.with_initial_state(current_state)

        try:
            model = gp.Model("planar_box_pushing_minlp")
        except gp.GurobiError as exc:
            raise MinlpSolveError(
                "Could not create a Gurobi model. Install/configure a valid "
                f"Gurobi license before selecting this backend: {exc.message}"
            ) from exc

        cfg = self.config
        system = self.system
        h = cfg.horizon
        force_max = system.force_scale * cfg.max_normal_force
        torque_max = force_max * (system.max_contact_radius + system.pusher_radius)
        m = cfg.big_m

        model.Params.OutputFlag = 0
        model.Params.NonConvex = 2
        model.Params.TimeLimit = cfg.solver_time_limit

        state_lb = [*cfg.position_lower, cfg.theta_lower, *cfg.position_lower]
        state_ub = [*cfg.position_upper, cfg.theta_upper, *cfg.position_upper]
        X = {
            (i, k): model.addVar(
                lb=state_lb[i], ub=state_ub[i], name=f"X[{i},{k}]"
            )
            for i in range(5)
            for k in range(h + 1)
        }
        U = {
            (i, k): model.addVar(
                lb=-cfg.max_pusher_speed,
                ub=cfg.max_pusher_speed,
                name=f"U[{i},{k}]",
            )
            for i in range(2)
            for k in range(h)
        }
        F = {
            (i, k): model.addVar(
                lb=-force_max, ub=force_max, name=f"F_B[{i},{k}]"
            )
            for i in range(2)
            for k in range(h)
        }
        TAU = {
            k: model.addVar(lb=-torque_max, ub=torque_max, name=f"tau_B[{k}]")
            for k in range(h)
        }
        COS = {
            k: model.addVar(lb=-1.0, ub=1.0, name=f"cos_theta[{k}]")
            for k in range(h + 1)
        }
        SIN = {
            k: model.addVar(lb=-1.0, ub=1.0, name=f"sin_theta[{k}]")
            for k in range(h + 1)
        }
        P_BP = {
            (i, k): model.addVar(
                lb=-m, ub=m, name=f"pusher_position_B[{i},{k}]"
            )
            for i in range(2)
            for k in range(h + 1)
        }
        Z = {
            (mode, k): model.addVar(vtype=GRB.BINARY, name=f"mode[{mode},{k}]")
            for mode in range(NUM_MODES)
            for k in range(h)
        }
        SWITCH = {
            k: model.addVar(lb=0.0, ub=1.0, name=f"switch[{k}]")
            for k in range(h - 1)
        }

        # The nonlinear definitions give Gurobi explicit rotation variables.
        for k in range(h + 1):
            model.addConstr(COS[k] == gp.nlfunc.cos(X[2, k]), name=f"cos[{k}]")
            model.addConstr(SIN[k] == gp.nlfunc.sin(X[2, k]), name=f"sin[{k}]")
            dx = X[3, k] - X[0, k]
            dy = X[4, k] - X[1, k]
            model.addQConstr(
                P_BP[0, k] == COS[k] * dx + SIN[k] * dy,
                name=f"pusher_body_x[{k}]",
            )
            model.addQConstr(
                P_BP[1, k] == -SIN[k] * dx + COS[k] * dy,
                name=f"pusher_body_y[{k}]",
            )

        initial = current_state.vector()
        for i in range(5):
            model.addConstr(X[i, 0] == float(initial[i]), name=f"initial[{i}]")

        for k in range(h):
            # The original ellipsoidal limit-surface dynamics, represented in
            # world coordinates using the explicit sine/cosine variables.
            model.addQConstr(
                X[0, k + 1] - X[0, k]
                == cfg.time_step
                / system.f_max**2
                * (COS[k] * F[0, k] - SIN[k] * F[1, k]),
                name=f"dynamics_x[{k}]",
            )
            model.addQConstr(
                X[1, k + 1] - X[1, k]
                == cfg.time_step
                / system.f_max**2
                * (SIN[k] * F[0, k] + COS[k] * F[1, k]),
                name=f"dynamics_y[{k}]",
            )
            model.addConstr(
                X[2, k + 1] - X[2, k]
                == cfg.time_step * TAU[k] / system.tau_max**2,
                name=f"dynamics_theta[{k}]",
            )
            for i in range(2):
                model.addConstr(
                    X[3 + i, k + 1] - X[3 + i, k]
                    == cfg.time_step * U[i, k],
                    name=f"pusher_dynamics[{i},{k}]",
                )
            model.addQConstr(
                U[0, k] * U[0, k] + U[1, k] * U[1, k]
                <= cfg.max_pusher_speed**2,
                name=f"pusher_speed[{k}]",
            )
            model.addConstr(
                gp.quicksum(Z[mode, k] for mode in range(NUM_MODES)) == 1,
                name=f"one_mode[{k}]",
            )

            contact_active = gp.quicksum(
                Z[face, k] for face in range(NUM_CONTACT_MODES)
            )
            for i in range(2):
                model.addConstr(
                    F[i, k] <= force_max * contact_active,
                    name=f"force_upper[{i},{k}]",
                )
                model.addConstr(
                    F[i, k] >= -force_max * contact_active,
                    name=f"force_lower[{i},{k}]",
                )
            model.addConstr(
                TAU[k] <= torque_max * contact_active, name=f"torque_upper[{k}]"
            )
            model.addConstr(
                TAU[k] >= -torque_max * contact_active, name=f"torque_lower[{k}]"
            )

            self._add_contact_constraints(
                model, k, X, F, TAU, P_BP, Z, force_max, m
            )
            self._add_free_space_constraints(model, k, P_BP, Z, m)

        self._add_transition_constraints(model, Z, SWITCH, previous_mode_idx)
        objective = self._add_objective(
            model, X, U, F, Z, SWITCH, goal, previous_mode_idx
        )
        model.setObjective(objective, GRB.MINIMIZE)
        self._set_warm_start(model, X, U, F, TAU, COS, SIN, P_BP, Z, SWITCH, warm_start)

        try:
            model.optimize()
        except gp.GurobiError as exc:
            raise MinlpSolveError(f"Gurobi failed while solving the MINLP: {exc.message}") from exc

        acceptable_statuses = {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
        if model.Status not in acceptable_statuses or model.SolCount == 0:
            raise MinlpSolveError(
                "Gurobi failed to produce a feasible MINLP solution: "
                f"status={model.Status}, solution_count={model.SolCount}"
            )
        return self._get_solution(model, X, U, F, TAU, Z)

    def _add_contact_constraints(
        self, model, k, X, F, TAU, P_BP, Z, force_max: float, m: float
    ) -> None:
        for face_idx in range(NUM_CONTACT_MODES):
            z = Z[face_idx, k]
            loc = PolytopeContactLocation(ContactLocation.FACE, face_idx)
            v1, v2 = self.geometry.get_proximate_vertices_from_location(loc)
            normal, tangent = self.geometry.get_norm_and_tang_vecs_from_location(loc)
            normal = normal.reshape(-1)
            tangent = tangent.reshape(-1)
            pbc_x = P_BP[0, k] + normal[0] * self.system.pusher_radius
            pbc_y = P_BP[1, k] + normal[1] * self.system.pusher_radius
            plane = normal[0] * (pbc_x - v1[0, 0]) + normal[1] * (
                pbc_y - v1[1, 0]
            )
            model.addConstr(plane <= m * (1 - z), name=f"face_upper[{face_idx},{k}]")
            model.addConstr(plane >= -m * (1 - z), name=f"face_lower[{face_idx},{k}]")

            tangent_coordinate = tangent[0] * pbc_x + tangent[1] * pbc_y
            t1 = float((tangent.reshape(1, 2) @ v1).item())
            t2 = float((tangent.reshape(1, 2) @ v2).item())
            tangent_lb, tangent_ub = min(t1, t2), max(t1, t2)
            model.addConstr(
                tangent_coordinate >= tangent_lb - m * (1 - z),
                name=f"face_segment_lower[{face_idx},{k}]",
            )
            model.addConstr(
                tangent_coordinate <= tangent_ub + m * (1 - z),
                name=f"face_segment_upper[{face_idx},{k}]",
            )

            normal_force = normal[0] * F[0, k] + normal[1] * F[1, k]
            friction_force = tangent[0] * F[0, k] + tangent[1] * F[1, k]
            model.addConstr(
                normal_force >= -m * (1 - z), name=f"normal_lower[{face_idx},{k}]"
            )
            model.addConstr(
                normal_force <= force_max + m * (1 - z),
                name=f"normal_upper[{face_idx},{k}]",
            )
            model.addConstr(
                friction_force
                <= self.system.friction_coeff_slider_pusher * normal_force
                + m * (1 - z),
                name=f"friction_upper[{face_idx},{k}]",
            )
            model.addConstr(
                -friction_force
                <= self.system.friction_coeff_slider_pusher * normal_force
                + m * (1 - z),
                name=f"friction_lower[{face_idx},{k}]",
            )

            torque = pbc_x * F[1, k] - pbc_y * F[0, k]
            model.addQConstr(
                TAU[k] - torque <= m * (1 - z), name=f"torque_upper[{face_idx},{k}]"
            )
            model.addQConstr(
                TAU[k] - torque >= -m * (1 - z), name=f"torque_lower[{face_idx},{k}]"
            )
            for i in range(2):
                displacement = P_BP[i, k + 1] - P_BP[i, k]
                model.addConstr(
                    displacement <= m * (1 - z), name=f"stick_upper[{face_idx},{i},{k}]"
                )
                model.addConstr(
                    displacement >= -m * (1 - z), name=f"stick_lower[{face_idx},{i},{k}]"
                )

    def _add_free_space_constraints(self, model, k, P_BP, Z, m: float) -> None:
        for region_idx in range(NUM_FREE_MODES):
            z = Z[NUM_CONTACT_MODES + region_idx, k]
            planes = [self.geometry.faces[region_idx]] + list(
                self.geometry.get_planes_for_collision_free_region(region_idx)
            )
            for plane_idx, plane in enumerate(planes):
                clearance = self.system.pusher_radius if plane_idx == 0 else 0.0
                a = plane.a.reshape(-1)
                b = float(plane.b.item())
                for state_idx in (k, k + 1):
                    model.addConstr(
                        a[0] * P_BP[0, state_idx] + a[1] * P_BP[1, state_idx]
                        - b
                        >= clearance - m * (1 - z),
                        name=f"free[{region_idx},{plane_idx},{state_idx},{k}]",
                    )

    def _add_transition_constraints(self, model, Z, SWITCH, previous_mode: Optional[int]) -> None:
        h = self.config.horizon
        for k in range(h - 1):
            for first_face in range(NUM_CONTACT_MODES):
                for second_face in range(NUM_CONTACT_MODES):
                    if first_face != second_face:
                        model.addConstr(
                            Z[first_face, k] + Z[second_face, k + 1] <= 1,
                            name=f"no_direct_face_switch[{first_face},{second_face},{k}]",
                        )
            for mode_idx in range(NUM_MODES):
                model.addConstr(
                    SWITCH[k] >= Z[mode_idx, k] - Z[mode_idx, k + 1],
                    name=f"switch[{mode_idx},{k}]",
                )

        if previous_mode is not None:
            allowed_contact = (
                previous_mode
                if previous_mode < NUM_CONTACT_MODES
                else previous_mode - NUM_CONTACT_MODES
            )
            for face_idx in range(NUM_CONTACT_MODES):
                if face_idx != allowed_contact:
                    model.addConstr(
                        Z[face_idx, 0] == 0, name=f"carried_mode[{face_idx}]"
                    )
        if self.config.required_initial_mode is not None:
            model.addConstr(
                Z[self.config.required_initial_mode, 0] == 1,
                name="required_initial_mode",
            )

    def _add_objective(
        self, model, X, U, F, Z, SWITCH, goal: PlanarPushingGoal, previous_mode
    ):
        h = self.config.horizon
        cfg = self.config
        goal_xy = goal.slider.pos().reshape(-1)
        cos_terminal_error = model.addVar(lb=-1.0, ub=1.0, name="cos_terminal_error")
        model.addConstr(
            cos_terminal_error == gp.nlfunc.cos(X[2, h] - goal.slider.theta),
            name="terminal_orientation_error",
        )
        objective = (
            cfg.terminal_position_weight
            * ((X[0, h] - goal_xy[0]) * (X[0, h] - goal_xy[0])
               + (X[1, h] - goal_xy[1]) * (X[1, h] - goal_xy[1]))
            + cfg.terminal_orientation_weight * (1 - cos_terminal_error)
        )
        for k in range(h):
            objective += cfg.running_position_weight * (
                (X[0, k] - goal_xy[0]) * (X[0, k] - goal_xy[0])
                + (X[1, k] - goal_xy[1]) * (X[1, k] - goal_xy[1])
            )
            objective += cfg.pusher_motion_weight * (
                U[0, k] * U[0, k] + U[1, k] * U[1, k]
            )
            objective += cfg.force_weight * (F[0, k] * F[0, k] + F[1, k] * F[1, k])
        objective += cfg.mode_switch_weight * gp.quicksum(SWITCH.values())
        if previous_mode is not None:
            objective += cfg.mode_switch_weight * (1 - Z[previous_mode, 0])
        return objective

    def _set_warm_start(self, model, X, U, F, TAU, COS, SIN, P_BP, Z, SWITCH, warm) -> None:
        h = self.config.horizon
        expected_shapes = {
            "states": (5, h + 1),
            "pusher_velocities": (2, h),
            "contact_forces_B": (2, h),
            "contact_torques_B": (h,),
            "modes": (NUM_MODES, h),
        }
        invalid_shapes = {
            name: np.asarray(getattr(warm, name)).shape
            for name, expected in expected_shapes.items()
            if np.asarray(getattr(warm, name)).shape != expected
        }
        if invalid_shapes:
            raise ValueError(
                "Gurobi warm start has shapes incompatible with the horizon: "
                f"{invalid_shapes}; expected {expected_shapes}"
            )
        for k in range(h + 1):
            theta = warm.states[2, k]
            for i in range(5):
                X[i, k].Start = float(warm.states[i, k])
            COS[k].Start = float(np.cos(theta))
            SIN[k].Start = float(np.sin(theta))
            rotation = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            pusher_body = rotation.T @ (warm.states[3:5, k] - warm.states[0:2, k])
            for i in range(2):
                P_BP[i, k].Start = float(pusher_body[i])
        for k in range(h):
            for i in range(2):
                U[i, k].Start = float(warm.pusher_velocities[i, k])
                F[i, k].Start = float(warm.contact_forces_B[i, k])
            TAU[k].Start = float(warm.contact_torques_B[k])
            for mode_idx in range(NUM_MODES):
                Z[mode_idx, k].Start = float(warm.modes[mode_idx, k] > 0.5)
        for k in range(h - 1):
            previous = int(np.argmax(warm.modes[:, k]))
            current = int(np.argmax(warm.modes[:, k + 1]))
            SWITCH[k].Start = float(previous != current)
        model.update()

    def _get_solution(self, model, X, U, F, TAU, Z) -> MinlpMpcSolution:
        h = self.config.horizon
        states = np.array([[X[i, k].X for k in range(h + 1)] for i in range(5)])
        velocities = np.array([[U[i, k].X for k in range(h)] for i in range(2)])
        forces = np.array([[F[i, k].X for k in range(h)] for i in range(2)])
        torques = np.array([TAU[k].X for k in range(h)])
        modes = np.array(
            [[round(Z[mode_idx, k].X) for k in range(h)] for mode_idx in range(NUM_MODES)]
        )
        try:
            mip_gap = float(model.MIPGap)
        except gp.GurobiError:
            mip_gap = np.nan
        stats = {
            "solver": "gurobi",
            "status": int(model.Status),
            "solution_count": int(model.SolCount),
            "runtime": float(model.Runtime),
            "mip_gap": mip_gap,
        }
        return MinlpMpcSolution(
            states=states,
            pusher_velocities=velocities,
            contact_forces_B=forces,
            contact_torques_B=torques,
            modes=modes,
            mode_indices=np.argmax(modes, axis=0).astype(int),
            objective=float(model.ObjVal),
            solver_stats=stats,
        )
