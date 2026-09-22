# RESULTS_LEDGER

Status: no experiments recorded under the current method specification.

Use this file only for results that are traceable to a concrete repository commit and result artifact.

## Entry template

### YYYY-MM-DD — experiment name

- Method-spec commit:
- Code commit:
- Branch:
- Experiment script:
- Configuration file / command:
- Physical scenes:
- Seeds:
- Online evaluation budget:
- Result files:
- Inner-solver validation status:
- Key numerical results:
- Failures / infeasible calls:
- Interpretation allowed by the evidence:
- Claims that are NOT supported:
- Notes:

---

## Current status

### 2026-09-22 — pre-outer core sensing/CRB/inner validation

- Method-spec commit: `4fa60f82b6aa0c0561169a2fce5af30c35014f97`
- Code commit: `aeb76111b44ff7127f6786403b1e70eaf65cbfa6`
- Branch: `paper-method-sync`
- Experiment script: `scripts/run_core_validation.py`
- Configuration file / command: `.venv/Scripts/python.exe scripts/run_core_validation.py --output results/generated/core_validation_20260922`
- Physical scenes: deterministic synthetic fixed-array validation cases only; three fixed arrays are recorded in `summary.json`
- Seeds: master seed `20260922`; inner seeds `20260932`, `20260933`, `20260934`
- Online evaluation budget: not applicable; outer search was not executed
- Result files: `results/generated/core_validation_20260922/summary.json`, `results/generated/core_validation_20260922/inner_convergence.csv`
- Inner-solver validation status: passed all implemented unit and integration checks (`12 passed`)
- Key numerical results:
  - left-pseudoinverse relative error: `5.2954e-16`
  - `G^H G = Q^T kron I` relative error: `4.7717e-17`
  - processed-noise Monte Carlo covariance relative error: `7.1194e-3` with `200000` samples
  - maximum full/Woodbury inverse, FIM, and CRB relative errors: `1.9485e-13`, `1.0274e-12`, `3.7101e-13`
  - 64-dimensional covariance inverse timing: full `5.8460e-4` s, Woodbury `2.0950e-4` s per call (`2.7905x` speedup); Woodbury was slower for the 9- and 16-dimensional cases because fixed overhead dominated
  - maximum/median analytical-gradient directional-check relative errors: `3.7818e-8` / `7.2550e-9`
  - three fixed-array accepted CRB decreases: `3.4397%`, `3.4824%`, `2.2804%`
  - all accepted inner points feasible within the declared `2e-6` numerical tolerance; accepted true CRB sequences were monotonically nonincreasing
  - maximum rank-one desired-power error: `1.1102e-16`; maximum recorded Q reconstruction relative error: `6.7934e-16`; all recovered SINRs exceeded their targets
- Failures / infeasible calls: none in this deterministic validation set
- Interpretation allowed by the evidence: the implemented full and Woodbury paths are numerically equivalent for these cases; the analytical CRB gradient passes directional checks; the implemented accepted inner iterates remain feasible and their true CRB is nonincreasing; the specified rank-one recovery preserves feasibility in these cases
- Claims that are NOT supported: global optimum, local optimum, joint KKT convergence, general runtime superiority at every matrix size, final paper performance, or any outer-search/prior/GP claim
- Notes: The conventional unnormalised half-wavelength ULA convention `a_0[m]=exp(j*pi*m*sin(theta))` was used because `METHOD_SPEC.md` does not state a steering-vector normalization. This convention is isolated in `arrays.py` and should be explicitly frozen before final experiments if the paper intends a different normalization.

---

Current status: core pre-outer validation is recorded above. No final outer-search result has been generated under the current method specification.
