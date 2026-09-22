import numpy as np

from isac_method.sensing import (
    covariance_from_precoder,
    left_pseudoinverse,
    processed_noise_covariance,
    sample_processed_noise,
    sensing_operator,
)


def _rectangular_precoder(seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mt, kc = 3, 2
    w = rng.standard_normal((mt, kc + mt)) + 1j * rng.standard_normal((mt, kc + mt))
    return w.astype(np.complex128)


def test_left_inverse_and_gram_identity() -> None:
    w = _rectangular_precoder()
    mr = 2
    q = covariance_from_precoder(w)
    g = sensing_operator(w, mr)
    v = left_pseudoinverse(g)
    np.testing.assert_allclose(v @ g, np.eye(mt_mr := w.shape[0] * mr), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        g.conj().T @ g,
        np.kron(q.T, np.eye(mr)),
        rtol=2e-13,
        atol=2e-13,
    )
    assert g.shape == ((w.shape[1] * mr), mt_mr)
    assert v.shape == (mt_mr, w.shape[1] * mr)


def test_processed_noise_monte_carlo() -> None:
    w = _rectangular_precoder(17)
    mr, n0 = 2, 0.35
    q = covariance_from_precoder(w)
    g = sensing_operator(w, mr)
    v = left_pseudoinverse(g)
    samples = sample_processed_noise(v, n0, 80_000, np.random.default_rng(91))
    empirical = samples @ samples.conj().T / samples.shape[1]
    theory = processed_noise_covariance(q, mr, n0)
    relative_error = np.linalg.norm(empirical - theory, "fro") / np.linalg.norm(theory, "fro")
    assert relative_error < 0.015

