import numpy as np
import pytest

from isac_method.arrays import (
    candidate_ula_steering,
    make_selection,
    manifold,
    sum_coarray_positions,
    sum_coarray_steering,
)


def test_selection_and_sum_dimensions() -> None:
    selection = make_selection(8, [0, 3, 7], [1, 2, 5, 6])
    theta = 0.23
    a, a_dot = sum_coarray_steering(theta, selection)
    assert selection.tx_matrix.shape == (3, 8)
    assert selection.rx_matrix.shape == (4, 8)
    assert a.shape == (12,)
    assert a_dot.shape == (12,)
    assert np.array_equal(sum_coarray_positions(selection), np.unique(
        selection.tx_indices[:, None] + selection.rx_indices[None, :]
    ))


def test_sum_steering_matches_selected_kronecker_product() -> None:
    selection = make_selection(7, [0, 2, 6], [1, 4])
    theta = -0.31
    a0 = candidate_ula_steering(theta, 7)
    actual, _ = sum_coarray_steering(theta, selection)
    expected = np.kron(selection.tx_matrix @ a0, selection.rx_matrix @ a0)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_steering_derivative_matches_central_difference() -> None:
    selection = make_selection(9, [0, 4, 8], [1, 2, 6])
    theta = 0.18
    step = 1e-7
    _, analytic = sum_coarray_steering(theta, selection)
    forward, _ = sum_coarray_steering(theta + step, selection)
    backward, _ = sum_coarray_steering(theta - step, selection)
    numerical = (forward - backward) / (2.0 * step)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-8, atol=2e-8)


def test_manifold_dimension() -> None:
    selection = make_selection(8, [0, 3, 7], [1, 5])
    a, a_dot = manifold([-0.2, 0.1, 0.4], selection)
    assert a.shape == (6, 3)
    assert a_dot.shape == (6, 3)


def test_overlapping_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        make_selection(6, [0, 2], [2, 5])

