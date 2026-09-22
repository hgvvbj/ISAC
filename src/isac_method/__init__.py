"""Numerical implementation of the current paper method specification.

Outer physical-prior/GP search is intentionally absent until its six feature
definitions are frozen in ``METHOD_SPEC.md``.
"""

from .arrays import (
    ArraySelection,
    candidate_ula_steering,
    make_selection,
    sum_coarray_positions,
    sum_coarray_steering,
)
from .crb import CRBResult, stochastic_crb
from .gradients import crb_gradient, directional_derivative
from .sensing import (
    left_pseudoinverse,
    processed_noise_covariance,
    sensing_operator,
)

__all__ = [
    "ArraySelection",
    "CRBResult",
    "candidate_ula_steering",
    "crb_gradient",
    "directional_derivative",
    "left_pseudoinverse",
    "make_selection",
    "processed_noise_covariance",
    "sensing_operator",
    "stochastic_crb",
    "sum_coarray_positions",
    "sum_coarray_steering",
]
