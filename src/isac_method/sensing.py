"""Rectangular sensing operator and left-pseudoinverse noise model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]


def _hermitian(matrix: ComplexArray) -> ComplexArray:
    return (0.5 * (matrix + matrix.conj().T)).astype(np.complex128)


def assert_positive_definite(matrix: ComplexArray, *, tolerance: float = 1e-11) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not np.allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-12):
        raise ValueError("matrix must be Hermitian")
    smallest = float(np.linalg.eigvalsh(_hermitian(matrix)).min())
    if smallest <= tolerance:
        raise ValueError(f"matrix must be positive definite; lambda_min={smallest:.3e}")


def covariance_from_precoder(w: ComplexArray) -> ComplexArray:
    if w.ndim != 2 or w.shape[0] > w.shape[1]:
        raise ValueError("W must be a rectangular matrix with at least as many columns as rows")
    q = _hermitian(w @ w.conj().T)
    assert_positive_definite(q)
    return q


def sensing_operator(w: ComplexArray, mr: int) -> ComplexArray:
    """Construct ``G = W.T kron I_Mr``."""

    if mr <= 0:
        raise ValueError("mr must be positive")
    if w.ndim != 2:
        raise ValueError("W must be two-dimensional")
    return np.kron(w.T, np.eye(mr, dtype=np.complex128)).astype(np.complex128)


def left_pseudoinverse(g: ComplexArray) -> ComplexArray:
    """Compute the full-column-rank left inverse without explicitly inverting G^H G."""

    if g.ndim != 2 or g.shape[0] < g.shape[1]:
        raise ValueError("G must be tall or square")
    gram = _hermitian(g.conj().T @ g)
    assert_positive_definite(gram)
    return np.linalg.solve(gram, g.conj().T).astype(np.complex128)


def processed_noise_covariance(q: ComplexArray, mr: int, n0: float) -> ComplexArray:
    """Return ``N0 * ((Q.T)^-1 kron I_Mr)``."""

    if n0 <= 0.0:
        raise ValueError("n0 must be positive")
    if mr <= 0:
        raise ValueError("mr must be positive")
    q = np.asarray(q, dtype=np.complex128)
    assert_positive_definite(q)
    q_t_inverse = np.linalg.solve(q.T, np.eye(q.shape[0], dtype=np.complex128))
    return _hermitian(n0 * np.kron(q_t_inverse, np.eye(mr, dtype=np.complex128)))


def inverse_processed_noise_covariance(q: ComplexArray, mr: int, n0: float) -> ComplexArray:
    """Return the exact inverse ``(1/N0) * (Q.T kron I_Mr)``."""

    if n0 <= 0.0:
        raise ValueError("n0 must be positive")
    q = np.asarray(q, dtype=np.complex128)
    assert_positive_definite(q)
    return _hermitian(np.kron(q.T / n0, np.eye(mr, dtype=np.complex128)))


def sample_processed_noise(
    v: ComplexArray,
    n0: float,
    sample_count: int,
    rng: np.random.Generator,
) -> ComplexArray:
    """Generate columns of white proper-complex noise and apply ``V``."""

    if sample_count <= 1 or n0 <= 0.0:
        raise ValueError("sample_count must exceed one and n0 must be positive")
    dimension = v.shape[1]
    white = np.sqrt(n0 / 2.0) * (
        rng.standard_normal((dimension, sample_count))
        + 1j * rng.standard_normal((dimension, sample_count))
    )
    return (v @ white).astype(np.complex128)

