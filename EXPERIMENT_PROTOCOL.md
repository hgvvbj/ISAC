# EXPERIMENT_PROTOCOL

Status: draft protocol. No final experiment has been run under this specification yet.

The purpose of this file is to prevent method changes, tuning leakage, hidden failures, and mixing results from incompatible algorithm versions.

## 1. General rules

1. The implementation must follow `METHOD_SPEC.md`.
2. Any change to the mathematical method must first be approved in the paper-method discussion and committed to `METHOD_SPEC.md`.
3. Experimental hyperparameters may be tuned only on designated development scenes.
4. Validation and final holdout scenes must not be used to choose method components.
5. Once the algorithm version is frozen, final experiments must use the frozen version without scene-specific manual changes.
6. All failed inner-solver calls consume the online evaluation budget.
7. Communication-infeasible and numerical-failure evaluations must not be assigned synthetic CRB values.
8. All reported results must be traceable to:
   - repository commit,
   - configuration,
   - random seed,
   - result file.

## 2. Required method families

At minimum compare:

- historical-prior + residual GP + EI,
- no-prior GP + EI,
- frozen prior only,
- random search,
- one structural heuristic baseline,
- suitable combinatorial-optimization baseline when computationally feasible.

All online methods must share:

- the same initial evaluated configurations,
- the same inner solver,
- the same true-evaluation budget,
- the same feasibility handling.

## 3. Inner-solver validation

Before large outer experiments, verify:

- left-pseudoinverse identity,
- processed colored-noise covariance,
- analytical/implemented CRB consistency,
- low-dimensional Woodbury CRB equals full-dimensional CRB,
- gradient checks,
- SINR feasibility,
- total power,
- minimum eigenvalue of Q,
- recovered W has full row rank,
- rank-one communication recovery,
- sensing covariance reconstruction,
- accepted true-CRB monotonicity.

Do not claim more convergence than directly supported.

## 4. Scene split

Before tuning the outer algorithm, predefine:

- development physical scenes,
- validation physical scenes,
- final holdout physical scenes.

The exact scene definitions and random seeds will be inserted before experiments begin.

## 5. Historical-prior data

Historical data are offline cost and must be reported separately from online evaluation cost.

For each historical physical scene:

- generate structurally admissible arrays,
- run the same frozen inner solver used by the target-scene evaluations,
- keep feasible true-CRB labels,
- convert feasible CRBs within that scene to percentile utilities in [0,1].

Feature standardization statistics are computed from the historical training set only and then frozen.

## 6. Outer online evaluation

The exact online budget and candidate-pool size are not yet frozen.

Current design intent:

- shared initial true evaluations,
- fixed total online evaluation budget,
- each outer iteration builds a finite structurally valid nonrepeated candidate pool,
- candidate pool may combine global sampling, local mutation around good arrays, and physics-diverse candidates,
- cheap features/prior/GP/EI may be evaluated on the whole candidate pool,
- only the selected candidate calls the expensive inner solver.

Hyperparameters such as candidate-pool size, allocation ratios, ridge regularization, GP numerical noise, kernel optimization iterations, and online budget are implementation/experimental parameters and should be tuned only on development scenes.

## 7. Metrics

Report by independent physical scene, not by treating repeated random runs as independent scenes.

Required metrics include:

- final best true CRB,
- relative CRB improvement,
- paired win/tie/loss summaries,
- uncertainty/confidence intervals when appropriate,
- infeasible-call rate,
- online true-evaluation count,
- offline prior-labeling cost,
- online runtime,
- low-dimensional CRB numerical accuracy.

Repeated seeds within one physical scene must be clearly distinguished from independent scenes.

## 8. Ablations

At minimum test:

- full prior + residual GP,
- prior only,
- GP without historical prior,
- random search.

Additional ablations may isolate physical-feature groups or candidate-pool components if space permits.

The claim that the residual GP adds value beyond the frozen prior is allowed only if supported by the ablation results.

## 9. Version locking

Before final holdout evaluation, record in this file:

- method-spec commit,
- implementation commit,
- frozen hyperparameters,
- development scenes used for tuning,
- validation scenes,
- final holdout scenes,
- seed list,
- result directory convention.

After this lock, do not alter the algorithm based on final holdout outcomes.
