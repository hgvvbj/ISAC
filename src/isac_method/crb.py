"""Full and Woodbury-exact stochastic angular CRB implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .sensing import (
    assert_positive_definite,
    inverse_processed_noise_covariance,
    processed_noise_covariance,
)


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


class NumericalCRBError(RuntimeError):
    """Raised when a CRB evaluation is singular or numerically invalid."""


@dataclass(frozen=True)
class CRBResult:
    objective: float
    ry: ComplexArray
    ry_inverse: ComplexArray
    fim: RealArray
    j_theta_theta: RealArray
    j_theta_power: RealArray
    j_power_power: RealArray
    j_effective: RealArray


def _hermitian(matrix: ComplexArray) -> ComplexArray:
    return (0.5 * (matrix + matrix.conj().T)).astype(np.complex128)


def observation_covariance(
    q: ComplexArray,
    a: ComplexArray,
    powers: NDArray[np.floating] | list[float],
    mr: int,
    n0: float,
) -> ComplexArray:
    powers_array = np.asarray(powers, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != powers_array.size:
        raise ValueError("A must have one column per target power")
    if np.any(powers_array <= 0.0):
        raise ValueError("target powers must be positive")
    expected = q.shape[0] * mr
    if a.shape[0] != expected:
        raise ValueError("A row count must equal M_t*M_r")
    signal = (a * powers_array[None, :]) @ a.conj().T
    return _hermitian(signal + processed_noise_covariance(q, mr, n0))


def full_inverse(ry: ComplexArray) -> ComplexArray:
    assert_positive_definite(ry)
    result = np.linalg.solve(ry, np.eye(ry.shape[0], dtype=np.complex128))
    return _hermitian(result)


def woodbury_inverse(
    q: ComplexArray,
    a: ComplexArray,
    powers: NDArray[np.floating] | list[float],
    mr: int,
    n0: float,
) -> ComplexArray:
    """Compute ``R_y^-1`` exactly via a K_s-dimensional Woodbury solve."""

    powers_array = np.asarray(powers, dtype=np.float64)
    if np.any(powers_array <= 0.0):
        raise ValueError("target powers must be positive")
    rn_inverse = inverse_processed_noise_covariance(q, mr, n0)
    ap = a * np.sqrt(powers_array)[None, :]
    b = rn_inverse @ ap
    middle = _hermitian(np.eye(ap.shape[1], dtype=np.complex128) + ap.conj().T @ b)
    correction = b @ np.linalg.solve(middle, b.conj().T)
    return _hermitian(rn_inverse - correction)


def covariance_derivatives(
    a: ComplexArray,
    a_dot: ComplexArray,
    powers: NDArray[np.floating] | list[float],
) -> list[ComplexArray]:
    """Return derivatives ordered as all angles followed by all powers."""

    powers_array = np.asarray(powers, dtype=np.float64)
    if a.shape != a_dot.shape or a.shape[1] != powers_array.size:
        raise ValueError("A, A_dot, and powers have incompatible dimensions")
    derivatives: list[ComplexArray] = []
    for index, power in enumerate(powers_array):
        ak = a[:, index]
        dak = a_dot[:, index]
        derivatives.append(
            _hermitian(power * (np.outer(dak, ak.conj()) + np.outer(ak, dak.conj())))
        )
    for index in range(powers_array.size):
        ak = a[:, index]
        derivatives.append(_hermitian(np.outer(ak, ak.conj())))
    return derivatives


def slepian_bangs_fim(
    ry_inverse: ComplexArray,
    derivatives: list[ComplexArray],
    snapshots: int,
) -> RealArray:
    if snapshots <= 0:
        raise ValueError("snapshots must be positive")
    products = [ry_inverse @ derivative for derivative in derivatives]
    parameter_count = len(products)
    fim = np.empty((parameter_count, parameter_count), dtype=np.float64)
    for row in range(parameter_count):
        for column in range(row, parameter_count):
            value = snapshots * float(np.real(np.trace(products[row] @ products[column])))
            fim[row, column] = value
            fim[column, row] = value
    return 0.5 * (fim + fim.T)


def effective_angle_information(fim: RealArray, target_count: int) -> tuple[RealArray, RealArray, RealArray, RealArray]:
    if fim.shape != (2 * target_count, 2 * target_count):
        raise ValueError("FIM dimension does not match angle/power parameterisation")
    jtt = fim[:target_count, :target_count]
    jtp = fim[:target_count, target_count:]
    jpp = fim[target_count:, target_count:]
    try:
        schur_term = jtp @ np.linalg.solve(jpp, jtp.T)
    except np.linalg.LinAlgError as exc:
        raise NumericalCRBError("power nuisance-information block is singular") from exc
    jeff = 0.5 * ((jtt - schur_term) + (jtt - schur_term).T)
    smallest = float(np.linalg.eigvalsh(jeff).min())
    scale = max(1.0, float(np.linalg.norm(jeff, ord=2)))
    if smallest <= 1e-11 * scale:
        raise NumericalCRBError(
            f"effective angle information is not numerically positive definite; "
            f"lambda_min={smallest:.3e}, scale={scale:.3e}"
        )
    return jtt, jtp, jpp, jeff


def stochastic_crb(
    q: ComplexArray,
    a: ComplexArray,
    a_dot: ComplexArray,
    powers: NDArray[np.floating] | list[float],
    mr: int,
    n0: float,
    snapshots: int,
    *,
    inverse_method: Literal["full", "woodbury"] = "woodbury",
) -> CRBResult:
    """Evaluate ``tr(J_eff^-1)`` under the frozen stochastic target model."""

    ry = observation_covariance(q, a, powers, mr, n0)
    if inverse_method == "full":
        ry_inverse = full_inverse(ry)
    elif inverse_method == "woodbury":
        ry_inverse = woodbury_inverse(q, a, powers, mr, n0)
    else:
        raise ValueError(f"unknown inverse_method: {inverse_method}")
    derivatives = covariance_derivatives(a, a_dot, powers)
    fim = slepian_bangs_fim(ry_inverse, derivatives, snapshots)
    target_count = a.shape[1]
    jtt, jtp, jpp, jeff = effective_angle_information(fim, target_count)
    try:
        objective = float(np.trace(np.linalg.solve(jeff, np.eye(target_count))).real)
    except np.linalg.LinAlgError as exc:
        raise NumericalCRBError("effective angle information is singular") from exc
    if not np.isfinite(objective) or objective <= 0.0:
        raise NumericalCRBError(f"invalid CRB objective: {objective}")
    return CRBResult(
        objective=objective,
        ry=ry,
        ry_inverse=ry_inverse,
        fim=fim,
        j_theta_theta=jtt,
        j_theta_power=jtp,
        j_power_power=jpp,
        j_effective=jeff,
    )

