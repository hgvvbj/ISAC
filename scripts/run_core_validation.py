"""Run traceable validation of the frozen sensing/CRB/inner method.

This script intentionally contains no outer-search feature, prior, GP, or EI
implementation.  It writes JSON and CSV artifacts instead of relying on
terminal-only output.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np

from isac_method.arrays import make_selection, manifold, sum_coarray_positions
from isac_method.communication import covariance_sinrs
from isac_method.crb import full_inverse, observation_covariance, stochastic_crb, woodbury_inverse
from isac_method.gradients import crb_gradient, directional_derivative
from isac_method.inner_solver import InnerSolverConfig, solve_fixed_array
from isac_method.recovery import recover_precoders
from isac_method.sensing import (
    covariance_from_precoder,
    left_pseudoinverse,
    processed_noise_covariance,
    sample_processed_noise,
    sensing_operator,
)


METHOD_SPEC_COMMIT = "4fa60f82b6aa0c0561169a2fce5af30c35014f97"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected, "fro") / np.linalg.norm(expected, "fro"))


def _random_q(mt: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal((mt, mt)) + 1j * rng.standard_normal((mt, mt))
    return raw @ raw.conj().T + 0.6 * np.eye(mt)


def _selection_for_size(mt: int, mr: int, variant: int):
    candidate_size = mt + mr + 5
    rng = np.random.default_rng(10_000 + 97 * mt + 13 * variant)
    indices = rng.choice(candidate_size, size=mt + mr, replace=False)
    return make_selection(candidate_size, indices[:mt], indices[mt:])


def validate_dimensions_and_pseudoinverse(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    selection = make_selection(11, [0, 4, 7, 10], [1, 3, 6])
    a, a_dot = manifold(np.deg2rad([-25.0, 3.0, 29.0]), selection)
    kc = 3
    w = rng.standard_normal((selection.mt, kc + selection.mt)) + 1j * rng.standard_normal(
        (selection.mt, kc + selection.mt)
    )
    q = covariance_from_precoder(w)
    g = sensing_operator(w, selection.mr)
    v = left_pseudoinverse(g)
    identity_error = _relative_error(v @ g, np.eye(selection.mt * selection.mr))
    gram_error = _relative_error(
        g.conj().T @ g, np.kron(q.T, np.eye(selection.mr))
    )
    return {
        "candidate_size": selection.candidate_size,
        "mt": selection.mt,
        "mr": selection.mr,
        "kc": kc,
        "ks": a.shape[1],
        "selection_matrix_shapes": [list(selection.tx_matrix.shape), list(selection.rx_matrix.shape)],
        "manifold_shape": list(a.shape),
        "manifold_derivative_shape": list(a_dot.shape),
        "sum_positions": sum_coarray_positions(selection).tolist(),
        "distinct_sum_count": int(sum_coarray_positions(selection).size),
        "w_shape": list(w.shape),
        "g_shape": list(g.shape),
        "v_shape": list(v.shape),
        "left_inverse_relative_error": identity_error,
        "gram_identity_relative_error": gram_error,
        "passed": identity_error < 1e-11 and gram_error < 1e-11,
    }


def validate_noise(seed: int, sample_count: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mt, mr, kc, n0 = 4, 3, 2, 0.3
    w = rng.standard_normal((mt, kc + mt)) + 1j * rng.standard_normal((mt, kc + mt))
    q = covariance_from_precoder(w)
    g = sensing_operator(w, mr)
    v = left_pseudoinverse(g)
    samples = sample_processed_noise(v, n0, sample_count, rng)
    empirical = samples @ samples.conj().T / sample_count
    theory = processed_noise_covariance(q, mr, n0)
    error = _relative_error(empirical, theory)
    return {
        "sample_count": sample_count,
        "dimension": int(mt * mr),
        "n0": n0,
        "relative_covariance_error": error,
        "theory_trace": float(np.trace(theory).real),
        "empirical_trace": float(np.trace(empirical).real),
        "passed": error < 0.02,
    }


def validate_full_vs_woodbury(seed: int, repetitions: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cases: list[dict[str, Any]] = []
    dimensions = [(3, 3, 2), (4, 4, 3), (6, 5, 3), (8, 8, 3)]
    for variant, (mt, mr, ks) in enumerate(dimensions):
        selection = _selection_for_size(mt, mr, variant)
        base_angles = np.linspace(-0.55, 0.55, ks) + rng.uniform(-0.025, 0.025, ks)
        a, a_dot = manifold(base_angles, selection)
        q = _random_q(mt, rng)
        powers = rng.uniform(0.6, 1.4, ks)
        n0, snapshots = 0.25, 48
        ry = observation_covariance(q, a, powers, mr, n0)
        inverse_full = full_inverse(ry)
        inverse_reduced = woodbury_inverse(q, a, powers, mr, n0)
        full_result = stochastic_crb(
            q, a, a_dot, powers, mr, n0, snapshots, inverse_method="full"
        )
        reduced_result = stochastic_crb(
            q, a, a_dot, powers, mr, n0, snapshots, inverse_method="woodbury"
        )

        start = time.perf_counter()
        for _ in range(repetitions):
            full_inverse(ry)
        full_seconds = (time.perf_counter() - start) / repetitions
        start = time.perf_counter()
        for _ in range(repetitions):
            woodbury_inverse(q, a, powers, mr, n0)
        reduced_seconds = (time.perf_counter() - start) / repetitions

        inverse_error = _relative_error(inverse_reduced, inverse_full)
        fim_error = float(
            np.linalg.norm(reduced_result.fim - full_result.fim, "fro")
            / np.linalg.norm(full_result.fim, "fro")
        )
        crb_error = abs(reduced_result.objective - full_result.objective) / abs(
            full_result.objective
        )
        cases.append(
            {
                "mt": mt,
                "mr": mr,
                "ks": ks,
                "observation_dimension": mt * mr,
                "inverse_relative_error": inverse_error,
                "fim_relative_error": fim_error,
                "crb_relative_error": crb_error,
                "full_inverse_seconds": full_seconds,
                "woodbury_inverse_seconds": reduced_seconds,
                "inverse_speedup_full_over_woodbury": full_seconds / reduced_seconds,
                "full_crb": full_result.objective,
                "woodbury_crb": reduced_result.objective,
                "passed": max(inverse_error, fim_error, crb_error) < 2e-8,
            }
        )
    return {
        "repetitions_per_case": repetitions,
        "cases": cases,
        "max_inverse_relative_error": max(case["inverse_relative_error"] for case in cases),
        "max_fim_relative_error": max(case["fim_relative_error"] for case in cases),
        "max_crb_relative_error": max(case["crb_relative_error"] for case in cases),
        "passed": all(case["passed"] for case in cases),
    }


def validate_gradient(seed: int, case_count: int, directions_per_case: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    errors: list[float] = []
    records: list[dict[str, Any]] = []
    for case in range(case_count):
        selection = _selection_for_size(4, 3, case)
        angles = np.deg2rad([-28.0, 5.0, 32.0]) + rng.uniform(-0.01, 0.01, 3)
        a, a_dot = manifold(angles, selection)
        q = _random_q(4, rng)
        powers = rng.uniform(0.7, 1.3, 3)
        objective, gradient = crb_gradient(q, a, a_dot, powers, selection.mr, 0.25, 40)
        for direction_index in range(directions_per_case):
            raw = rng.standard_normal(q.shape) + 1j * rng.standard_normal(q.shape)
            direction = 0.5 * (raw + raw.conj().T)
            direction /= np.linalg.norm(direction, "fro")
            step = 2e-5 * min(1.0, float(np.linalg.eigvalsh(q).min()))
            plus = stochastic_crb(
                q + step * direction,
                a,
                a_dot,
                powers,
                selection.mr,
                0.25,
                40,
                inverse_method="full",
            ).objective
            minus = stochastic_crb(
                q - step * direction,
                a,
                a_dot,
                powers,
                selection.mr,
                0.25,
                40,
                inverse_method="full",
            ).objective
            finite_difference = (plus - minus) / (2.0 * step)
            analytic = directional_derivative(gradient, direction)
            relative_error = abs(analytic - finite_difference) / max(
                1e-14, abs(analytic), abs(finite_difference)
            )
            errors.append(relative_error)
            records.append(
                {
                    "case": case,
                    "direction": direction_index,
                    "objective": objective,
                    "analytic": analytic,
                    "finite_difference": finite_difference,
                    "relative_error": relative_error,
                }
            )
    return {
        "case_count": case_count,
        "directions_per_case": directions_per_case,
        "records": records,
        "maximum_relative_error": max(errors),
        "median_relative_error": float(np.median(errors)),
        "passed": max(errors) < 5e-5,
    }


def _inner_selection(index: int):
    patterns = [
        ([0, 3, 7, 11], [1, 4, 8, 10]),
        ([0, 4, 8, 11], [1, 3, 7, 10]),
        ([0, 2, 7, 11], [1, 4, 8, 10]),
    ]
    tx, rx = patterns[index]
    return make_selection(12, tx, rx)


def validate_inner_and_recovery(seed: int) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for index in range(3):
        rng = np.random.default_rng(seed + index)
        selection = _inner_selection(index)
        angles = np.deg2rad([-30.0, 2.0, 33.0]) + rng.uniform(-0.015, 0.015, 3)
        a, a_dot = manifold(angles, selection)
        channels = rng.standard_normal((selection.mt, 3)) + 1j * rng.standard_normal(
            (selection.mt, 3)
        )
        channels /= np.linalg.norm(channels, axis=0, keepdims=True)
        targets = np.array([1.8, 2.2, 2.0])
        noises = np.array([0.05, 0.05, 0.05])
        config = InnerSolverConfig(
            p_max=4.0,
            epsilon_q=0.05,
            initial_rho=1e-3,
            max_iterations=8,
            relative_tolerance=1e-6,
            acceptance_tolerance=1e-10,
        )
        result = solve_fixed_array(
            a,
            a_dot,
            [1.0, 0.85, 0.7],
            selection.mr,
            0.25,
            40,
            channels,
            targets,
            noises,
            config,
        )
        before_sinr = covariance_sinrs(result.covariances, channels, noises)
        recovery = recover_precoders(result.q, result.covariances, channels, noises)
        accepted = [item for item in result.history if item.accepted]
        accepted_crbs = [item.true_crb for item in accepted]
        monotone = all(
            after <= before + config.acceptance_tolerance * max(abs(before), 1e-15)
            for before, after in zip(accepted_crbs, accepted_crbs[1:])
        )
        accepted_feasible = all(
            item.minimum_sinr_margin >= -config.feasibility_tolerance
            and item.total_power <= config.p_max + config.feasibility_tolerance
            and item.lambda_min_q >= config.epsilon_q - config.feasibility_tolerance
            and item.q_residual_min_eigenvalue >= -config.feasibility_tolerance
            for item in accepted
        )
        scenario_passed = bool(
            monotone
            and accepted_feasible
            and np.all(recovery.sinrs >= targets - config.feasibility_tolerance)
            and recovery.covariance_relative_error < 1e-7
            and recovery.dominance_min_eigenvalues.min() >= -2e-7
            and recovery.sensing_min_eigenvalue >= -2e-7
        )
        scenarios.append(
            {
                "scenario": index,
                "seed": seed + index,
                "tx_indices": selection.tx_indices.tolist(),
                "rx_indices": selection.rx_indices.tolist(),
                "status": result.status,
                "initial_crb": accepted_crbs[0],
                "final_crb": result.objective,
                "relative_crb_decrease": (accepted_crbs[0] - result.objective) / accepted_crbs[0],
                "accepted_count": len(accepted),
                "monotone_accepted_crb": monotone,
                "all_accepted_feasible": accepted_feasible,
                "history": result.history_as_dicts(),
                "sinr_targets": targets.tolist(),
                "sinr_before_recovery": before_sinr.tolist(),
                "sinr_after_recovery": recovery.sinrs.tolist(),
                "q_reconstruction_relative_error": recovery.covariance_relative_error,
                "maximum_desired_power_error": float(recovery.desired_power_errors.max()),
                "minimum_rk_minus_rhat_eigenvalue": float(
                    recovery.dominance_min_eigenvalues.min()
                ),
                "sensing_covariance_min_eigenvalue": recovery.sensing_min_eigenvalue,
                "passed": scenario_passed,
            }
        )
    return {"scenarios": scenarios, "passed": all(item["passed"] for item in scenarios)}


def _write_inner_csv(path: Path, scenarios: list[dict[str, Any]]) -> None:
    fields = [
        "scenario",
        "iteration",
        "accepted",
        "true_crb",
        "surrogate_objective",
        "minimum_sinr_margin",
        "total_power",
        "lambda_min_q",
        "q_residual_min_eigenvalue",
        "step_size",
        "rho",
        "solver_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scenario in scenarios:
            for row in scenario["history"]:
                writer.writerow({"scenario": scenario["scenario"], **row})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/generated/core_validation"))
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--noise-samples", type=int, default=200_000)
    parser.add_argument("--timing-repetitions", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "method_spec_commit": METHOD_SPEC_COMMIT,
        "code_commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "seed": args.seed,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cvxpy": cp.__version__,
            "solvers": cp.installed_solvers(),
            "platform": platform.platform(),
        },
        "outer_search_executed": False,
    }
    report["dimensions_and_pseudoinverse"] = validate_dimensions_and_pseudoinverse(args.seed)
    report["colored_noise"] = validate_noise(args.seed + 1, args.noise_samples)
    report["full_vs_woodbury"] = validate_full_vs_woodbury(
        args.seed + 2, args.timing_repetitions
    )
    report["gradient"] = validate_gradient(args.seed + 3, 4, 5)
    report["inner_and_recovery"] = validate_inner_and_recovery(args.seed + 10)
    report["all_passed"] = all(
        report[key]["passed"]
        for key in (
            "dimensions_and_pseudoinverse",
            "colored_noise",
            "full_vs_woodbury",
            "gradient",
            "inner_and_recovery",
        )
    )

    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _write_inner_csv(
        args.output / "inner_convergence.csv", report["inner_and_recovery"]["scenarios"]
    )
    print(json.dumps({"output": str(args.output), "all_passed": report["all_passed"]}))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

