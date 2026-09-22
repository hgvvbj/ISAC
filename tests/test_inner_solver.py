import numpy as np
import pytest

from isac_method.arrays import make_selection, manifold
from isac_method.communication import covariance_feasibility_metrics, is_feasible
from isac_method.inner_solver import (
    CommunicationInfeasible,
    InnerSolverConfig,
    feasible_initialization,
    solve_fixed_array,
)
from isac_method.recovery import recover_precoders


def _fixed_array_problem(seed: int = 12):
    rng = np.random.default_rng(seed)
    selection = make_selection(9, [0, 4, 8], [1, 3, 7])
    a, a_dot = manifold(np.deg2rad([-24.0, 28.0]), selection)
    channels = rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))
    channels /= np.linalg.norm(channels, axis=0, keepdims=True)
    targets = np.array([2.0, 2.5])
    noises = np.array([0.08, 0.08])
    config = InnerSolverConfig(
        p_max=3.0,
        epsilon_q=0.08,
        initial_rho=1e-3,
        max_iterations=6,
        relative_tolerance=1e-7,
    )
    return selection, a, a_dot, channels, targets, noises, config


def test_feasible_initialization_and_rank_one_recovery() -> None:
    _, _, _, channels, targets, noises, config = _fixed_array_problem()
    q, covariances, status = feasible_initialization(channels, targets, noises, config)
    assert status in {"optimal", "optimal_inaccurate"}
    metrics = covariance_feasibility_metrics(
        q, covariances, channels, targets, noises, config.p_max, config.epsilon_q
    )
    assert is_feasible(metrics, tolerance=config.feasibility_tolerance)
    recovery = recover_precoders(q, covariances, channels, noises)
    assert np.all(recovery.sinrs >= targets - 2e-6)
    assert recovery.covariance_relative_error < 2e-9
    assert recovery.desired_power_errors.max() < 2e-9
    assert recovery.dominance_min_eigenvalues.min() > -2e-7
    assert recovery.sensing_min_eigenvalue > -2e-7
    assert recovery.ws.shape == (channels.shape[0], channels.shape[0])
    full_precoder = np.column_stack((recovery.wc, recovery.ws))
    assert full_precoder.shape == (channels.shape[0], channels.shape[1] + channels.shape[0])
    assert np.linalg.matrix_rank(full_precoder) == channels.shape[0]


def test_communication_infeasibility_is_reported_without_fake_crb() -> None:
    channels = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    config = InnerSolverConfig(p_max=0.2, epsilon_q=0.01)
    with pytest.raises(CommunicationInfeasible):
        feasible_initialization(
            channels,
            sinr_targets=[10.0, 10.0],
            noise_powers=[1.0, 1.0],
            config=config,
        )


def test_inner_solver_preserves_feasibility_and_monotone_accepted_crb() -> None:
    selection, a, a_dot, channels, targets, noises, config = _fixed_array_problem(27)
    result = solve_fixed_array(
        a,
        a_dot,
        [1.0, 0.8],
        selection.mr,
        0.25,
        32,
        channels,
        targets,
        noises,
        config,
    )
    accepted = [item for item in result.history if item.accepted]
    objectives = np.asarray([item.true_crb for item in accepted])
    assert accepted
    assert np.all(np.diff(objectives) <= 1e-10 * np.maximum(1e-15, np.abs(objectives[:-1])))
    assert all(item.minimum_sinr_margin >= -config.feasibility_tolerance for item in accepted)
    assert all(item.total_power <= config.p_max + config.feasibility_tolerance for item in accepted)
    assert all(item.lambda_min_q >= config.epsilon_q - config.feasibility_tolerance for item in accepted)
    assert all(item.q_residual_min_eigenvalue >= -config.feasibility_tolerance for item in accepted)

    recovery = recover_precoders(result.q, result.covariances, channels, noises)
    assert np.all(recovery.sinrs >= targets - 2e-6)
    assert recovery.covariance_relative_error < 2e-8
