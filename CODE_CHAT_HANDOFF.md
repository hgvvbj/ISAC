# CODE_CHAT_HANDOFF

Use this file when a separate ChatGPT/Codex session is responsible for implementation.

## Repository and branch

- Repository: `hgvvbj/ISAC`
- Working branch: `paper-method-sync`
- Treat `METHOD_SPEC.md` as the current mathematical method definition.
- Treat `EXPERIMENT_PROTOCOL.md` as the experiment-governance document.
- Record only traceable experimental outcomes in `RESULTS_LEDGER.md`.

## Role of the code session

The code session implements and validates the method already approved in the paper-method discussion. It must not silently redesign the method.

Before changing code:

1. Read `METHOD_SPEC.md` completely.
2. Read `EXPERIMENT_PROTOCOL.md` completely.
3. Check the latest commit on `paper-method-sync`.
4. Identify which requested components are frozen and which are still marked unresolved.

If a requested implementation conflicts with `METHOD_SPEC.md`, stop and report the conflict instead of inventing a replacement model.

## Current implementation priority

Implement in this order unless the paper-method discussion explicitly changes it:

1. Core array/steering and Sum-coarray utilities.
2. Communication SINR model.
3. Sensing covariance and left-pseudoinverse model.
4. Full-dimensional stochastic CRB reference implementation.
5. Woodbury-based exact low-dimensional CRB implementation and equivalence tests.
6. Fixed-array covariance-domain inner solver.
7. Rank-one communication recovery and sensing-covariance reconstruction.
8. Outer-search scaffolding.
9. Historical physical prior, residual GP, and EI only after the six physical-feature formulas are frozen in `METHOD_SPEC.md`.

## Important unresolved item

The six outer physical-feature concepts are selected, but their exact mathematical normalizations are NOT yet frozen. Do not invent or tune feature formulas before `METHOD_SPEC.md` is updated with exact definitions.

The six concepts are:

1. distinct Sum-position ratio,
2. weakest target angular sensitivity,
3. maximum target steering correlation,
4. weakest user log-channel strength,
5. normalized inverse condition number of the user-channel matrix,
6. maximum communication-user / sensing-target directional coupling.

## Coding rules

- Prefer clear, testable modules over one large script.
- Use deterministic seeds for tests.
- Keep full-dimensional and reduced-dimensional CRB implementations separate so they can be cross-checked.
- Add explicit dimension and Hermitian/PSD assertions where useful during development.
- Do not hide infeasible cases or numerical failures.
- Do not assign fake CRB values to failed evaluations.
- Do not claim convergence or optimality beyond what `METHOD_SPEC.md` allows.
- Do not mix code/results from older incompatible formulations.

## Required validation before outer experiments

At minimum verify:

- steering/selection dimensions,
- Sum-coarray construction,
- left-pseudoinverse identity,
- processed colored-noise covariance,
- full vs reduced CRB equivalence,
- gradient correctness for the inner objective,
- SINR constraints,
- total power,
- minimum eigenvalue of Q,
- feasibility preservation,
- accepted true-CRB monotonicity,
- rank-one communication recovery,
- sensing-covariance reconstruction.

## Commit and result discipline

Every meaningful implementation change should be committed with a concise message.

Every experiment intended for paper use must record:

- method-spec commit,
- code commit,
- script/configuration,
- scenes,
- seeds,
- budget,
- result artifact path,
- failures/infeasible calls,
- numerical summary.

Do not write a paper claim into `RESULTS_LEDGER.md` unless the result supports it directly.
