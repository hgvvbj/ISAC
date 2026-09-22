"""Rank-one communication recovery and residual sensing factorisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .communication import beamformer_sinrs


@dataclass(frozen=True)
class RecoveryResult:
    wc: np.ndarray
    ws: np.ndarray
    recovered_covariances: list[np.ndarray]
    sensing_covariance: np.ndarray
    sinrs: np.ndarray
    covariance_relative_error: float
    desired_power_errors: np.ndarray
    dominance_min_eigenvalues: np.ndarray
    sensing_min_eigenvalue: float


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def recover_precoders(
    q: np.ndarray,
    covariances: list[np.ndarray],
    channels: np.ndarray,
    noise_powers: np.ndarray | list[float],
    *,
    eigenvalue_tolerance: float = 2e-7,
) -> RecoveryResult:
    """Apply the rank-one recovery stated in ``METHOD_SPEC.md``."""

    if channels.shape[1] != len(covariances):
        raise ValueError("one covariance is required per communication user")
    beamformers: list[np.ndarray] = []
    recovered: list[np.ndarray] = []
    desired_errors: list[float] = []
    dominance_eigenvalues: list[float] = []
    for covariance, channel in zip(covariances, channels.T):
        denominator = float(np.real(np.vdot(channel, covariance @ channel)))
        if denominator <= 0.0:
            raise ValueError("rank-one recovery requires positive desired signal power")
        beamformer = covariance @ channel / np.sqrt(denominator)
        rank_one = np.outer(beamformer, beamformer.conj())
        beamformers.append(beamformer)
        recovered.append(_hermitian(rank_one))
        recovered_power = float(np.real(np.vdot(channel, rank_one @ channel)))
        desired_errors.append(abs(recovered_power - denominator))
        difference = _hermitian(covariance - rank_one)
        dominance_eigenvalues.append(float(np.linalg.eigvalsh(difference).min()))

    wc = np.column_stack(beamformers)
    sensing_covariance = _hermitian(q - sum(recovered, np.zeros_like(q)))
    eigenvalues, eigenvectors = np.linalg.eigh(sensing_covariance)
    if float(eigenvalues.min()) < -eigenvalue_tolerance:
        raise ValueError(
            f"residual sensing covariance is not PSD; lambda_min={eigenvalues.min():.3e}"
        )
    clipped = np.maximum(eigenvalues, 0.0)
    ws = eigenvectors @ np.diag(np.sqrt(clipped))
    reconstructed = wc @ wc.conj().T + ws @ ws.conj().T
    relative_error = float(np.linalg.norm(reconstructed - q, "fro") / np.linalg.norm(q, "fro"))
    return RecoveryResult(
        wc=wc,
        ws=ws,
        recovered_covariances=recovered,
        sensing_covariance=sensing_covariance,
        sinrs=beamformer_sinrs(wc, channels, noise_powers),
        covariance_relative_error=relative_error,
        desired_power_errors=np.asarray(desired_errors, dtype=np.float64),
        dominance_min_eigenvalues=np.asarray(dominance_eigenvalues, dtype=np.float64),
        sensing_min_eigenvalue=float(eigenvalues.min()),
    )

