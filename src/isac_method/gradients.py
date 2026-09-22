"""Analytical Hermitian gradient of the stochastic CRB objective."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .crb import CRBResult, covariance_derivatives, stochastic_crb


ComplexArray = NDArray[np.complex128]


def _hermitian(matrix: ComplexArray) -> ComplexArray:
    return (0.5 * (matrix + matrix.conj().T)).astype(np.complex128)


def _objective_gradient_wrt_fim(result: CRBResult, target_count: int) -> NDArray[np.float64]:
    """Return the real symmetric derivative ``d f / d J``.

    This differentiates both the Schur complement that eliminates nuisance
    powers and ``tr(J_eff^-1)``.  The returned matrix is defined by
    ``df = tr(K.T @ dJ)`` for symmetric FIM perturbations.
    """

    identity = np.eye(target_count, dtype=np.float64)
    e_inverse = np.linalg.solve(result.j_effective, identity)
    h = -(e_inverse @ e_inverse)
    b = result.j_theta_power
    c_inverse = np.linalg.solve(result.j_power_power, identity)

    k = np.zeros_like(result.fim)
    k[:target_count, :target_count] = h
    k_tp = -h @ b @ c_inverse
    k[:target_count, target_count:] = k_tp
    k[target_count:, :target_count] = k_tp.T
    k[target_count:, target_count:] = c_inverse @ b.T @ h @ b @ c_inverse
    return 0.5 * (k + k.T)


def _partial_trace_rx(matrix: ComplexArray, mt: int, mr: int) -> ComplexArray:
    """Map ``dK kron I_Mr`` back to its ``M_t x M_t`` factor."""

    if matrix.shape != (mt * mr, mt * mr):
        raise ValueError("matrix dimension is incompatible with M_t and M_r")
    result = np.empty((mt, mt), dtype=np.complex128)
    for row in range(mt):
        for column in range(mt):
            block = matrix[row * mr : (row + 1) * mr, column * mr : (column + 1) * mr]
            result[row, column] = np.trace(block)
    return _hermitian(result)


def crb_gradient(
    q: ComplexArray,
    a: ComplexArray,
    a_dot: ComplexArray,
    powers: NDArray[np.floating] | list[float],
    mr: int,
    n0: float,
    snapshots: int,
) -> tuple[float, ComplexArray]:
    """Return ``f(Q)`` and its Hermitian gradient.

    The convention is

    ``df = Re tr(gradient^H dQ)``

    for every Hermitian perturbation ``dQ``.  The forward inverse is the exact
    Woodbury evaluation, while the reverse calculation differentiates the
    mathematically equivalent full covariance expression.
    """

    q = np.asarray(q, dtype=np.complex128)
    result = stochastic_crb(
        q,
        a,
        a_dot,
        powers,
        mr,
        n0,
        snapshots,
        inverse_method="woodbury",
    )
    derivatives = covariance_derivatives(a, a_dot, powers)
    target_count = a.shape[1]
    k_fim = _objective_gradient_wrt_fim(result, target_count)
    x = result.ry_inverse

    gradient_x = np.zeros_like(x)
    for row, derivative_row in enumerate(derivatives):
        for column, derivative_column in enumerate(derivatives):
            coefficient = k_fim[row, column]
            if coefficient == 0.0:
                continue
            gradient_x += (
                snapshots
                * coefficient
                * (derivative_row @ x @ derivative_column + derivative_column @ x @ derivative_row)
            )
    gradient_x = _hermitian(gradient_x)

    # X = R_y^-1, so dX = -X dR_y X.
    gradient_ry = _hermitian(-x @ gradient_x @ x)

    # R_n = N0 * (S^-1 kron I), S = Q.T.  The partial trace is
    # the adjoint of K -> K kron I.
    mt = q.shape[0]
    partial = _partial_trace_rx(gradient_ry, mt, mr)
    s_inverse = np.linalg.solve(q.T, np.eye(mt, dtype=np.complex128))
    gradient_s = _hermitian(-n0 * s_inverse.conj().T @ partial @ s_inverse.conj().T)
    gradient_q = _hermitian(gradient_s.T)
    return result.objective, gradient_q


def directional_derivative(
    gradient: ComplexArray,
    direction: ComplexArray,
) -> float:
    """Evaluate ``Re tr(gradient^H direction)``."""

    return float(np.real(np.trace(gradient.conj().T @ direction)))

