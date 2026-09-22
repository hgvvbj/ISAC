import numpy as np

from isac_method.arrays import make_selection, manifold
from isac_method.crb import observation_covariance, stochastic_crb, woodbury_inverse


def _problem(seed: int):
    rng = np.random.default_rng(seed)
    selection = make_selection(12, [0, 4, 7, 11], [1, 3, 6])
    angles = np.deg2rad([-23.0, 4.0, 31.0]) + rng.uniform(-0.01, 0.01, 3)
    a, a_dot = manifold(angles, selection)
    z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    q = z @ z.conj().T + 0.8 * np.eye(4)
    powers = np.array([0.9, 1.3, 0.7])
    return selection, a, a_dot, q, powers


def test_full_and_woodbury_inverse_fim_and_crb_agree() -> None:
    for seed in range(5):
        selection, a, a_dot, q, powers = _problem(seed)
        full = stochastic_crb(q, a, a_dot, powers, selection.mr, 0.2, 48, inverse_method="full")
        reduced = stochastic_crb(
            q, a, a_dot, powers, selection.mr, 0.2, 48, inverse_method="woodbury"
        )
        np.testing.assert_allclose(reduced.ry_inverse, full.ry_inverse, rtol=2e-11, atol=2e-11)
        # Small off-diagonal FIM entries are obtained by subtractive complex
        # arithmetic after two independently formed inverses.  Their relative
        # error remains below 1e-9 even when the diagonal spans 1e6--1e7.
        np.testing.assert_allclose(reduced.fim, full.fim, rtol=1e-9, atol=1e-7)
        np.testing.assert_allclose(reduced.objective, full.objective, rtol=1e-9, atol=1e-13)


def test_woodbury_inverse_is_an_inverse() -> None:
    selection, a, _, q, powers = _problem(33)
    ry = observation_covariance(q, a, powers, selection.mr, 0.4)
    ry_inverse = woodbury_inverse(q, a, powers, selection.mr, 0.4)
    np.testing.assert_allclose(ry_inverse @ ry, np.eye(ry.shape[0]), rtol=5e-11, atol=5e-11)
