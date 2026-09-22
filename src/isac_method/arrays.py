"""Shared-candidate ULA selection and Sum co-array utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ArraySelection:
    """A disjoint Tx/Rx selection on a shared candidate ULA."""

    candidate_size: int
    tx_indices: NDArray[np.int64]
    rx_indices: NDArray[np.int64]
    tx_matrix: RealArray
    rx_matrix: RealArray

    @property
    def mt(self) -> int:
        return int(self.tx_indices.size)

    @property
    def mr(self) -> int:
        return int(self.rx_indices.size)


def _validated_indices(indices: NDArray[np.integer] | list[int], size: int) -> NDArray[np.int64]:
    result = np.asarray(indices, dtype=np.int64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("selection indices must be a nonempty one-dimensional array")
    if np.any(result < 0) or np.any(result >= size):
        raise ValueError("selection index is outside the candidate ULA")
    if np.unique(result).size != result.size:
        raise ValueError("selection indices must be unique")
    return np.sort(result)


def selection_matrix(indices: NDArray[np.integer] | list[int], size: int) -> RealArray:
    """Return S such that ``S @ a0`` keeps the requested candidate entries."""

    idx = _validated_indices(indices, size)
    return np.eye(size, dtype=np.float64)[idx]


def make_selection(
    candidate_size: int,
    tx_indices: NDArray[np.integer] | list[int],
    rx_indices: NDArray[np.integer] | list[int],
) -> ArraySelection:
    if candidate_size <= 1:
        raise ValueError("candidate_size must exceed one")
    tx = _validated_indices(tx_indices, candidate_size)
    rx = _validated_indices(rx_indices, candidate_size)
    if np.intersect1d(tx, rx).size:
        raise ValueError("Tx and Rx selections must be disjoint")
    return ArraySelection(
        candidate_size=candidate_size,
        tx_indices=tx,
        rx_indices=rx,
        tx_matrix=selection_matrix(tx, candidate_size),
        rx_matrix=selection_matrix(rx, candidate_size),
    )


def candidate_ula_steering(
    theta: float,
    candidate_size: int,
    *,
    derivative: bool = False,
) -> ComplexArray:
    """Half-wavelength ULA steering vector, using radians.

    The conventional unnormalised manifold is used:
    ``a0[m] = exp(1j*pi*m*sin(theta))`` for ``m=0,...,M-1``.
    With ``derivative=True`` this returns ``d a0 / d theta``.
    """

    if candidate_size <= 0:
        raise ValueError("candidate_size must be positive")
    positions = np.arange(candidate_size, dtype=np.float64)
    steering = np.exp(1j * np.pi * positions * np.sin(theta)).astype(np.complex128)
    if not derivative:
        return steering
    return (1j * np.pi * positions * np.cos(theta) * steering).astype(np.complex128)


def selected_steering(
    theta: float,
    selection: ArraySelection,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    """Return ``a_t, a_r, adot_t, adot_r``."""

    a0 = candidate_ula_steering(theta, selection.candidate_size)
    da0 = candidate_ula_steering(theta, selection.candidate_size, derivative=True)
    at = (selection.tx_matrix @ a0).astype(np.complex128)
    ar = (selection.rx_matrix @ a0).astype(np.complex128)
    dat = (selection.tx_matrix @ da0).astype(np.complex128)
    dar = (selection.rx_matrix @ da0).astype(np.complex128)
    return at, ar, dat, dar


def sum_coarray_steering(theta: float, selection: ArraySelection) -> tuple[ComplexArray, ComplexArray]:
    """Return ``a_t kron a_r`` and its angle derivative."""

    at, ar, dat, dar = selected_steering(theta, selection)
    a = np.kron(at, ar).astype(np.complex128)
    adot = (np.kron(dat, ar) + np.kron(at, dar)).astype(np.complex128)
    expected = selection.mt * selection.mr
    if a.shape != (expected,) or adot.shape != (expected,):
        raise AssertionError("unexpected Sum co-array steering dimension")
    return a, adot


def manifold(
    angles: NDArray[np.floating] | list[float],
    selection: ArraySelection,
) -> tuple[ComplexArray, ComplexArray]:
    """Return the Sum co-array manifold ``A`` and derivative matrix ``A_dot``."""

    values = [sum_coarray_steering(float(theta), selection) for theta in angles]
    if not values:
        raise ValueError("at least one target angle is required")
    return (
        np.column_stack([value[0] for value in values]).astype(np.complex128),
        np.column_stack([value[1] for value in values]).astype(np.complex128),
    )


def sum_coarray_positions(selection: ArraySelection) -> NDArray[np.int64]:
    """Return sorted distinct integer Sum positions of the selected Tx/Rx pair."""

    sums = selection.tx_indices[:, None] + selection.rx_indices[None, :]
    return np.unique(sums).astype(np.int64)

