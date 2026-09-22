"""Feasible fixed-array successive covariance solver.

The solver only supports the conclusions in ``METHOD_SPEC.md``: accepted
iterates remain feasible (up to declared numerical tolerance) and accepted
true CRB values are monotonically nonincreasing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from .communication import (
    FeasibilityMetrics,
    covariance_feasibility_metrics,
    is_feasible,
)
from .crb import NumericalCRBError, stochastic_crb
from .gradients import crb_gradient


ComplexArray = NDArray[np.complex128]


class InnerSolverError(RuntimeError):
    pass


class CommunicationInfeasible(InnerSolverError):
    pass


@dataclass(frozen=True)
class InnerSolverConfig:
    p_max: float
    epsilon_q: float
    initial_rho: float = 1.0
    max_iterations: int = 12
    relative_tolerance: float = 1e-5
    line_search_shrink: float = 0.5
    minimum_step: float = 1e-4
    acceptance_tolerance: float = 1e-10
    feasibility_tolerance: float = 2e-6
    solver: str = "CLARABEL"
    solver_max_iterations: int = 500


@dataclass(frozen=True)
class InnerIteration:
    iteration: int
    accepted: bool
    true_crb: float
    surrogate_objective: float
    minimum_sinr_margin: float
    total_power: float
    lambda_min_q: float
    q_residual_min_eigenvalue: float
    step_size: float
    rho: float
    solver_status: str


@dataclass(frozen=True)
class InnerResult:
    q: ComplexArray
    covariances: list[ComplexArray]
    objective: float
    status: str
    history: list[InnerIteration]

    def history_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.history]


def _hermitian(matrix: ComplexArray) -> ComplexArray:
    return (0.5 * (matrix + matrix.conj().T)).astype(np.complex128)


def _validate_inputs(
    channels: ComplexArray,
    sinr_targets: NDArray[np.floating] | list[float],
    noise_powers: NDArray[np.floating] | list[float],
    config: InnerSolverConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    targets = np.asarray(sinr_targets, dtype=np.float64)
    noises = np.asarray(noise_powers, dtype=np.float64)
    if channels.ndim != 2 or channels.shape[1] == 0:
        raise ValueError("channels must have shape (M_t, K_c) with K_c > 0")
    if targets.shape != (channels.shape[1],) or noises.shape != targets.shape:
        raise ValueError("one SINR target and noise power are required per user")
    if np.any(targets <= 0.0) or np.any(noises <= 0.0):
        raise ValueError("SINR targets and noise powers must be positive")
    if config.p_max <= 0.0 or config.epsilon_q <= 0.0:
        raise ValueError("p_max and epsilon_q must be positive")
    if config.p_max <= channels.shape[0] * config.epsilon_q:
        raise ValueError("p_max must exceed M_t * epsilon_q")
    return targets, noises


def _constraints(
    q_variable: cp.Variable,
    r_variables: list[cp.Variable],
    channels: ComplexArray,
    targets: NDArray[np.float64],
    noises: NDArray[np.float64],
    config: InnerSolverConfig,
) -> list[cp.Constraint]:
    mt = channels.shape[0]
    total_r = sum(r_variables)
    constraints: list[cp.Constraint] = [
        q_variable >> config.epsilon_q * np.eye(mt),
        q_variable - total_r >> 0,
        cp.real(cp.trace(q_variable)) <= config.p_max,
    ]
    constraints.extend(r_variable >> 0 for r_variable in r_variables)
    for index, channel in enumerate(channels.T):
        h_outer = np.outer(channel, channel.conj())
        desired = cp.real(cp.trace(h_outer @ r_variables[index]))
        total_received = cp.real(cp.trace(h_outer @ total_r))
        constraints.append((1.0 + 1.0 / targets[index]) * desired - total_received >= noises[index])
    return constraints


def _solve(problem: cp.Problem, config: InnerSolverConfig) -> str:
    try:
        if config.solver.upper() == "CLARABEL":
            problem.solve(
                solver="CLARABEL",
                verbose=False,
                max_iter=config.solver_max_iterations,
                tol_gap_abs=1e-8,
                tol_gap_rel=1e-8,
                tol_feas=1e-8,
            )
        elif config.solver.upper() == "SCS":
            problem.solve(
                solver="SCS",
                verbose=False,
                max_iters=max(2_000, config.solver_max_iterations),
                eps=1e-6,
            )
        else:
            problem.solve(solver=config.solver, verbose=False)
    except cp.error.SolverError as exc:
        raise InnerSolverError(f"convex solver failed: {exc}") from exc
    return str(problem.status)


def feasible_initialization(
    channels: ComplexArray,
    sinr_targets: NDArray[np.floating] | list[float],
    noise_powers: NDArray[np.floating] | list[float],
    config: InnerSolverConfig,
) -> tuple[ComplexArray, list[ComplexArray], str]:
    """Solve a lifted minimum-total-covariance feasibility SDP.

    Remaining power is added isotropically to ``Q`` as sensing covariance;
    ``R_k`` are unchanged, so all communication inequalities are preserved.
    """

    targets, noises = _validate_inputs(channels, sinr_targets, noise_powers, config)
    mt, user_count = channels.shape
    q_variable = cp.Variable((mt, mt), hermitian=True)
    r_variables = [cp.Variable((mt, mt), hermitian=True) for _ in range(user_count)]
    constraints = _constraints(q_variable, r_variables, channels, targets, noises, config)
    problem = cp.Problem(cp.Minimize(cp.real(cp.trace(q_variable))), constraints)
    status = _solve(problem, config)
    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or q_variable.value is None:
        raise CommunicationInfeasible(f"communication initialization status: {status}")

    q = _hermitian(np.asarray(q_variable.value, dtype=np.complex128))
    covariances = [_hermitian(np.asarray(variable.value, dtype=np.complex128)) for variable in r_variables]
    remaining = config.p_max - float(np.trace(q).real)
    if remaining < -config.feasibility_tolerance:
        raise CommunicationInfeasible("minimum feasible total covariance exceeds P_max")
    if remaining > 0.0:
        q = _hermitian(q + (remaining / mt) * np.eye(mt))
    metrics = covariance_feasibility_metrics(
        q, covariances, channels, targets, noises, config.p_max, config.epsilon_q
    )
    if not is_feasible(metrics, tolerance=config.feasibility_tolerance):
        raise InnerSolverError(f"solver returned a numerically infeasible initializer: {metrics}")
    return q, covariances, status


def _proximal_candidate(
    q_current: ComplexArray,
    gradient: ComplexArray,
    channels: ComplexArray,
    targets: NDArray[np.float64],
    noises: NDArray[np.float64],
    config: InnerSolverConfig,
    rho: float,
) -> tuple[ComplexArray, list[ComplexArray], str]:
    mt, user_count = channels.shape
    q_variable = cp.Variable((mt, mt), hermitian=True)
    r_variables = [cp.Variable((mt, mt), hermitian=True) for _ in range(user_count)]
    constraints = _constraints(q_variable, r_variables, channels, targets, noises, config)
    displacement = q_variable - q_current
    linear = cp.real(cp.trace(gradient.conj().T @ displacement))
    proximal = 0.5 * rho * cp.sum_squares(cp.abs(displacement))
    problem = cp.Problem(cp.Minimize(linear + proximal), constraints)
    status = _solve(problem, config)
    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or q_variable.value is None:
        raise InnerSolverError(f"proximal SDP status: {status}")
    q_candidate = _hermitian(np.asarray(q_variable.value, dtype=np.complex128))
    r_candidates = [_hermitian(np.asarray(variable.value, dtype=np.complex128)) for variable in r_variables]
    return q_candidate, r_candidates, status


def _surrogate_value(
    base_objective: float,
    gradient: ComplexArray,
    displacement: ComplexArray,
    rho: float,
) -> float:
    return float(
        base_objective
        + np.real(np.trace(gradient.conj().T @ displacement))
        + 0.5 * rho * np.linalg.norm(displacement, "fro") ** 2
    )


def solve_fixed_array(
    a: ComplexArray,
    a_dot: ComplexArray,
    powers: NDArray[np.floating] | list[float],
    mr: int,
    n0: float,
    snapshots: int,
    channels: ComplexArray,
    sinr_targets: NDArray[np.floating] | list[float],
    noise_powers: NDArray[np.floating] | list[float],
    config: InnerSolverConfig,
) -> InnerResult:
    """Run feasible proximal covariance updates for one fixed array."""

    targets, noises = _validate_inputs(channels, sinr_targets, noise_powers, config)
    q_current, r_current, initial_status = feasible_initialization(
        channels, targets, noises, config
    )
    objective_current = stochastic_crb(
        q_current, a, a_dot, powers, mr, n0, snapshots, inverse_method="woodbury"
    ).objective
    initial_metrics = covariance_feasibility_metrics(
        q_current, r_current, channels, targets, noises, config.p_max, config.epsilon_q
    )
    history = [
        InnerIteration(
            iteration=0,
            accepted=True,
            true_crb=objective_current,
            surrogate_objective=objective_current,
            minimum_sinr_margin=initial_metrics.minimum_sinr_margin,
            total_power=initial_metrics.total_power,
            lambda_min_q=initial_metrics.lambda_min_q,
            q_residual_min_eigenvalue=initial_metrics.q_residual_min_eigenvalue,
            step_size=0.0,
            rho=config.initial_rho,
            solver_status=initial_status,
        )
    ]
    rho = config.initial_rho
    termination = "maximum_iterations"

    for iteration in range(1, config.max_iterations + 1):
        _, gradient = crb_gradient(q_current, a, a_dot, powers, mr, n0, snapshots)
        q_candidate, r_candidates, solver_status = _proximal_candidate(
            q_current, gradient, channels, targets, noises, config, rho
        )
        direction_q = q_candidate - q_current
        direction_r = [candidate - current for candidate, current in zip(r_candidates, r_current)]
        if np.linalg.norm(direction_q, "fro") <= config.relative_tolerance * max(
            1.0, np.linalg.norm(q_current, "fro")
        ):
            termination = "small_candidate_step"
            break

        step = 1.0
        accepted = False
        last_metrics: FeasibilityMetrics | None = None
        last_objective = objective_current
        while step >= config.minimum_step:
            q_trial = _hermitian(q_current + step * direction_q)
            r_trial = [_hermitian(current + step * direction) for current, direction in zip(r_current, direction_r)]
            metrics = covariance_feasibility_metrics(
                q_trial, r_trial, channels, targets, noises, config.p_max, config.epsilon_q
            )
            last_metrics = metrics
            if not is_feasible(metrics, tolerance=config.feasibility_tolerance):
                step *= config.line_search_shrink
                continue
            try:
                trial_objective = stochastic_crb(
                    q_trial, a, a_dot, powers, mr, n0, snapshots, inverse_method="woodbury"
                ).objective
            except NumericalCRBError:
                step *= config.line_search_shrink
                continue
            last_objective = trial_objective
            allowed = objective_current + config.acceptance_tolerance * max(
                abs(objective_current), 1e-15
            )
            if trial_objective <= allowed:
                displacement = q_trial - q_current
                history.append(
                    InnerIteration(
                        iteration=iteration,
                        accepted=True,
                        true_crb=trial_objective,
                        surrogate_objective=_surrogate_value(
                            objective_current, gradient, displacement, rho
                        ),
                        minimum_sinr_margin=metrics.minimum_sinr_margin,
                        total_power=metrics.total_power,
                        lambda_min_q=metrics.lambda_min_q,
                        q_residual_min_eigenvalue=metrics.q_residual_min_eigenvalue,
                        step_size=step,
                        rho=rho,
                        solver_status=solver_status,
                    )
                )
                relative_change = abs(objective_current - trial_objective) / max(
                    abs(objective_current), 1e-15
                )
                q_current, r_current, objective_current = q_trial, r_trial, trial_objective
                accepted = True
                rho = max(config.initial_rho * 1e-4, rho * 0.7)
                if relative_change <= config.relative_tolerance:
                    termination = "small_objective_change"
                break
            step *= config.line_search_shrink

        if not accepted:
            metrics = last_metrics or covariance_feasibility_metrics(
                q_current, r_current, channels, targets, noises, config.p_max, config.epsilon_q
            )
            history.append(
                InnerIteration(
                    iteration=iteration,
                    accepted=False,
                    true_crb=last_objective,
                    surrogate_objective=_surrogate_value(
                        objective_current, gradient, step * direction_q, rho
                    ),
                    minimum_sinr_margin=metrics.minimum_sinr_margin,
                    total_power=metrics.total_power,
                    lambda_min_q=metrics.lambda_min_q,
                    q_residual_min_eigenvalue=metrics.q_residual_min_eigenvalue,
                    step_size=step,
                    rho=rho,
                    solver_status=solver_status,
                )
            )
            rho *= 10.0
            if rho > config.initial_rho * 1e8:
                termination = "line_search_failed"
                break
        elif termination == "small_objective_change":
            break

    return InnerResult(
        q=q_current,
        covariances=r_current,
        objective=objective_current,
        status=termination,
        history=history,
    )
