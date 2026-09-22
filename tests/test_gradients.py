import numpy as np

from isac_method.arrays import make_selection, manifold
from isac_method.crb import stochastic_crb
from isac_method.gradients import crb_gradient, directional_derivative


def _gradient_problem(seed: int):
    rng = np.random.default_rng(seed)
    selection = make_selection(10, [0, 3, 6, 9], [1, 4, 8])
    a, a_dot = manifold(np.deg2rad([-27.0, 6.0, 34.0]), selection)
    raw = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    q = raw @ raw.conj().T + 1.5 * np.eye(4)
    return rng, selection, a, a_dot, q, np.array([0.8, 1.2, 0.65])


def test_analytical_gradient_matches_directional_central_differences() -> None:
    for seed in (7, 19, 43):
        rng, selection, a, a_dot, q, powers = _gradient_problem(seed)
        objective, gradient = crb_gradient(q, a, a_dot, powers, selection.mr, 0.3, 40)
        reference = stochastic_crb(
            q, a, a_dot, powers, selection.mr, 0.3, 40, inverse_method="full"
        ).objective
        np.testing.assert_allclose(objective, reference, rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(gradient, gradient.conj().T, rtol=1e-12, atol=1e-12)

        for _ in range(4):
            raw_direction = rng.standard_normal(q.shape) + 1j * rng.standard_normal(q.shape)
            direction = 0.5 * (raw_direction + raw_direction.conj().T)
            direction /= np.linalg.norm(direction, "fro")
            step = 2e-5 * min(1.0, np.linalg.eigvalsh(q).min())
            forward = stochastic_crb(
                q + step * direction,
                a,
                a_dot,
                powers,
                selection.mr,
                0.3,
                40,
                inverse_method="full",
            ).objective
            backward = stochastic_crb(
                q - step * direction,
                a,
                a_dot,
                powers,
                selection.mr,
                0.3,
                40,
                inverse_method="full",
            ).objective
            finite_difference = (forward - backward) / (2.0 * step)
            analytic = directional_derivative(gradient, direction)
            denominator = max(1e-12, abs(finite_difference), abs(analytic))
            assert abs(finite_difference - analytic) / denominator < 2e-5

