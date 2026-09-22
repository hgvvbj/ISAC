"""Covariance-domain communication constraints and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class FeasibilityMetrics:
    sinr_margins: RealArray
    minimum_sinr_margin: float
    total_power: float
    power_margin: float
    lambda_min_q: float
    q_floor_margin: float
    q_residual_min_eigenvalue: float
    rk_min_eigenvalues: RealArray


def quadratic_form(channel: ComplexArray, covariance: ComplexArray) -> float:
    return float(np.real(np.vdot(channel, covariance @ channel)))


def lifted_sinr_margins(
    covariances: list[ComplexArray],
    channels: ComplexArray,
    sinr_targets: NDArray[np.floating] | list[float],
    noise_powers: NDArray[np.floating] | list[float],
) -> RealArray:
    """Return left-minus-right margins of all lifted SINR constraints."""

    targets = np.asarray(sinr_targets, dtype=np.float64)
    noises = np.asarray(noise_powers, dtype=np.float64)
    if channels.ndim != 2:
        raise ValueError("channels must have shape (M_t, K_c)")
    user_count = channels.shape[1]
    if len(covariances) != user_count or targets.shape != (user_count,) or noises.shape != (user_count,):
        raise ValueError("communication dimensions do not agree")
    if np.any(targets <= 0.0) or np.any(noises <= 0.0):
        raise ValueError("SINR targets and noise powers must be positive")
    total_communication = sum(covariances, np.zeros_like(covariances[0]))
    margins = np.empty(user_count, dtype=np.float64)
    for index in range(user_count):
        channel = channels[:, index]
        desired = quadratic_form(channel, covariances[index])
        total_received = quadratic_form(channel, total_communication)
        margins[index] = (1.0 + 1.0 / targets[index]) * desired - total_received - noises[index]
    return margins


def covariance_feasibility_metrics(
    q: ComplexArray,
    covariances: list[ComplexArray],
    channels: ComplexArray,
    sinr_targets: NDArray[np.floating] | list[float],
    noise_powers: NDArray[np.floating] | list[float],
    p_max: float,
    epsilon_q: float,
) -> FeasibilityMetrics:
    margins = lifted_sinr_margins(covariances, channels, sinr_targets, noise_powers)
    qh = 0.5 * (q + q.conj().T)
    residual = qh - sum(covariances, np.zeros_like(qh))
    power = float(np.real(np.trace(qh)))
    lambda_min = float(np.linalg.eigvalsh(qh).min())
    return FeasibilityMetrics(
        sinr_margins=margins,
        minimum_sinr_margin=float(margins.min()),
        total_power=power,
        power_margin=float(p_max - power),
        lambda_min_q=lambda_min,
        q_floor_margin=float(lambda_min - epsilon_q),
        q_residual_min_eigenvalue=float(np.linalg.eigvalsh(0.5 * (residual + residual.conj().T)).min()),
        rk_min_eigenvalues=np.asarray(
            [np.linalg.eigvalsh(0.5 * (rk + rk.conj().T)).min() for rk in covariances],
            dtype=np.float64,
        ),
    )


def is_feasible(metrics: FeasibilityMetrics, *, tolerance: float = 2e-6) -> bool:
    return bool(
        metrics.minimum_sinr_margin >= -tolerance
        and metrics.power_margin >= -tolerance
        and metrics.q_floor_margin >= -tolerance
        and metrics.q_residual_min_eigenvalue >= -tolerance
        and float(metrics.rk_min_eigenvalues.min()) >= -tolerance
    )


def beamformer_sinrs(
    wc: ComplexArray,
    channels: ComplexArray,
    noise_powers: NDArray[np.floating] | list[float],
) -> RealArray:
    """Evaluate communication SINRs after perfect sensing-waveform cancellation."""

    noises = np.asarray(noise_powers, dtype=np.float64)
    if wc.ndim != 2 or channels.shape != wc.shape or noises.shape != (wc.shape[1],):
        raise ValueError("W_c, channels, and noise powers have incompatible dimensions")
    received = channels.conj().T @ wc
    powers = np.abs(received) ** 2
    desired = np.diag(powers)
    interference = powers.sum(axis=1) - desired
    return (desired / (interference + noises)).astype(np.float64)

