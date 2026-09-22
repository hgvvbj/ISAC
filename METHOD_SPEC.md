# METHOD_SPEC

Status: current paper-method specification. Experimental implementation must follow this file unless a newer committed revision explicitly supersedes it.

## 1. Research problem

We study a monostatic far-field sparse MIMO-ISAC system with a common candidate ULA of M half-wavelength-spaced positions. Tx and Rx arrays are selected disjointly from the same candidate array.

- Tx/Rx binary selection vectors: `b_t, b_r in {0,1}^M`
- `1^T b_t = M_t`, `1^T b_r = M_r`
- `b_t + b_r <= 1` elementwise
- Only the Sum co-array is considered.
- Cheap structural screening uses the number of distinct Sum positions:
  `N_Sigma(b_t,b_r) >= K_s + 1`
- This is a screening condition only; it is not a sufficient identifiability/FIM-rank condition.

## 2. Array and waveform model

Let `S_t` and `S_r` be the Tx/Rx selection matrices, and let the candidate ULA steering vector be `a_0(theta)`. Then

`a_t(theta) = S_t a_0(theta)`,
`a_r(theta) = S_r a_0(theta)`.

Communication and sensing waveforms are kept separate in the statistical model.

- Communication waveform vector: `s_c(t) in C^{K_c}`
- Dedicated sensing basis waveform vector: `s_s(t) in C^{M_t}`
- Communication precoder: `W_c in C^{M_t x K_c}`
- Sensing precoder: `W_s in C^{M_t x M_t}`
- Overall precoder:
  `W = [W_c, W_s] in C^{M_t x (K_c+M_t)}`
- Joint waveform vector:
  `s(t) = [s_c^T(t), s_s^T(t)]^T`
- Transmitted signal:
  `x(t) = W s(t) = W_c s_c(t) + W_s s_s(t)`

The sensing-basis dimension is `M_t` so that an arbitrary PSD residual sensing covariance can be represented without an additional rank constraint.

## 3. Communication model

Dedicated sensing waveforms are assumed known at communication receivers and perfectly cancelled before data detection. Hence user-k SINR contains only communication multiuser interference and communication noise.

For user `k`, use `gamma_k(W_c)` as the SINR notation.

Do not impose `H_c W_s = 0`.

## 4. Sensing model

Continuous-time echo:

`y_s(t) = sum_k alpha_k a_r(theta_k) a_t^T(theta_k) W s(t-tau_k) + n_s(t)`.

For a range cell centered at reference delay `tau_0`, matched filtering uses the known transmitted waveform realization. For sensing processing, assume the finite processing block is normalized so that

`(1/T_p) int s(t)s^H(t) dt = I_{K_c+M_t}`.

For targets in the considered range cell, `tau_k approx tau_0`.

After matched filtering, define the effective coefficient
`q_k = sqrt(T_p) alpha_k`.

Vectorized sensing model:

`ybar_s = (W^T kron I_{M_r}) A q + nbar_s`,

where the Sum co-array manifold is
`A = [a(theta_1),...,a(theta_Ks)]`,
`a(theta_k) = a_t(theta_k) kron a_r(theta_k)`.

Define

`G = W^T kron I_{M_r}`,
`Q = W W^H`.

Because `W` is rectangular, `G` is non-square. Under `Q > 0`, `W` has full row rank and `G` has full column rank. Use the left pseudoinverse

`V = (G^H G)^{-1} G^H`,
with `V G = I`.

Equivalent sensing model:

`ytilde_s = A q + ntilde_s`.

Since
`G^H G = Q^T kron I_{M_r}`,

the processed noise covariance is

`R_n(Q) = N_0 ((Q^T)^{-1} kron I_{M_r})`.

## 5. Stochastic target model and CRB

For `L` independent sensing snapshots:

`q[i] ~ CN(0, diag(p))`.

The equivalent observation follows

`ytilde_s[i] ~ CN(0, R_y)`,

with

`R_y = A diag(p) A^H + N_0 ((Q^T)^{-1} kron I_{M_r})`.

Unknown parameters:

`theta = [theta_1,...,theta_Ks]^T`,
`p = [p_1,...,p_Ks]^T`,
`eta = [theta^T, p^T]^T`.

Target powers are nuisance parameters.

Use the zero-mean proper-complex Gaussian Slepian-Bangs FIM:

`J_uv = L Re tr(R_y^{-1} R_{y,u} R_y^{-1} R_{y,v})`.

Required derivatives:

`dR_y/dtheta_k = p_k (adot_k a_k^H + a_k adot_k^H)`,
`dR_y/dp_k = a_k a_k^H`.

Partition the FIM into angle/power blocks and eliminate `p` with the Schur complement:

`J_eff = J_tt - J_tp J_pp^{-1} J_pt`.

Sensing objective:

`f(Q) = tr(J_eff^{-1})`.

Do not claim a new CRB formula. The special structure is that the left-pseudoinverse processing makes the noise covariance depend on `Q`.

## 6. Original and covariance-domain optimization

Original physical-variable problem jointly optimizes

`b_t, b_r, W_c, W_s`

to minimize the sum angular CRB, subject to:

- binary/disjoint Tx/Rx selection,
- exact Tx/Rx cardinalities,
- per-user SINR constraints,
- total transmit power,
- stable full-row-rank condition:
  `W W^H >= epsilon_Q I`,
- structural Sum screening:
  `N_Sigma >= K_s+1`.

Define

`Q = W W^H`,
`R_k = w_{c,k} w_{c,k}^H`.

Then

`Q - sum_k R_k = W_s W_s^H >= 0`.

The lifted SINR constraint is

`(1+1/Gamma_k) h_k^H R_k h_k - h_k^H (sum_j R_j) h_k >= sigma_{c,k}^2`.

The covariance-domain problem uses variables

`b_t, b_r, Q, {R_k}`

with
`R_k >= 0`,
`rank(R_k)=1`,
`sum_k R_k <= Q`,
`tr(Q) <= P_max`,
`Q >= epsilon_Q I`.

## 7. Inner fixed-array solver

For fixed `b_t,b_r`, temporarily remove the rank-one constraints on `R_k`.

The feasible set is convex; the stochastic-CRB objective remains nonconvex.

### 7.1 Feasible initialization

1. Solve for the minimum communication power required to satisfy all SINR constraints.
2. If necessary, use a lifted covariance feasibility SDP.
3. Allocate remaining covariance to sensing while enforcing
   `Q >= epsilon_Q I`.
4. Infeasible communication cases are marked infeasible.

### 7.2 Exact low-dimensional CRB evaluation

Let

`R_n(Q) = N_0 ((Q^T)^{-1} kron I)`,
`A_p = A diag(sqrt(p))`.

Then

`R_y = R_n + A_p A_p^H`,

and

`R_n^{-1} = (1/N_0)(Q^T kron I)`.

Use Woodbury:

`R_y^{-1} = R_n^{-1} - R_n^{-1} A_p (I + A_p^H R_n^{-1} A_p)^{-1} A_p^H R_n^{-1}`.

This is exact, not an approximation. The target-dependent inversion is only `K_s x K_s`. FIM terms may be further evaluated through small Gram blocks built from `Z=[A,A_dot]`.

### 7.3 Successive covariance update

At feasible iterate `Q^(r)`, use the local proximal model

`fhat_r(Q) = f(Q^(r)) + <grad f(Q^(r)), Q-Q^(r)> + (rho_r/2)||Q-Q^(r)||_F^2`.

Solve the resulting convex SDP to obtain a candidate `Qhat,{Rhat_k}`.

Form a feasible line-search point using a convex combination of current and candidate variables. Accept only if the true CRB is nonincreasing.

Supported conclusion:
- accepted iterates remain feasible,
- accepted true CRB is monotonically nonincreasing.

Do NOT claim global optimality, local optimality, or joint KKT convergence unless separately proved.

### 7.4 Rank-one recovery

For each user:

`w_{c,k} = R_k h_k / sqrt(h_k^H R_k h_k)`,
`Rhat_k = w_{c,k} w_{c,k}^H`.

This preserves desired user signal power and satisfies `Rhat_k <= R_k`, so communication interference is not increased.

Residual sensing covariance:

`Q_s = Q - sum_k Rhat_k >= 0`.

Recover `W_s` from an eigendecomposition of `Q_s`.

The recovered precoders must satisfy

`W_c W_c^H + W_s W_s^H = Q`.

## 8. Outer sparse-array search

Array configuration:

`xi = [b_t^T, b_r^T]^T in {0,1}^{2M}`.

For each admissible `xi`, the frozen inner solver returns the optimized CRB `C(xi)`.

Only a limited number of array configurations may call the expensive inner solver.

### 8.1 Physical prior

Use six low-cost physical features, all computable before solving the inner problem:

1. distinct Sum-position ratio,
2. weakest target angular sensitivity,
3. maximum target steering correlation,
4. weakest user log-channel strength,
5. normalized inverse condition number of the user channel matrix,
6. maximum communication-user / sensing-target directional coupling.

Exact mathematical definitions of these six features are not yet frozen. They must be fixed before implementation and then kept unchanged during experiments.

Let

`phi(xi) in R^6`

be the feature vector.

Historical feasible arrays are converted, scene by scene, into percentile utilities in `[0,1]`. Standardize features using statistics from the historical training set only, and fit a frozen ridge-regression ranking prior

`m_0(xi) = omega_0 + omega^T phibar(xi)`.

### 8.2 New-scene utility and calibration

Using the initial feasible evaluations of the new scene, define the fixed reference CRB

`C_ref = median(initial feasible CRBs)`.

Online utility:

`y(xi) = -log(C(xi)/C_ref)`.

Calibrate the frozen prior with

`m_cal(xi) = kappa_0 + kappa_1 m_0(xi)`,
`kappa_1 >= 0`.

Residual:

`delta(xi) = y(xi) - m_cal(xi)`.

### 8.3 Residual GP

GP input:

`psi(xi) = [b_t^T, b_r^T, phibar^T(xi)]^T`.

Use grouped Matérn-5/2 structure over Tx mask, Rx mask, and physical-feature groups.

If the residual GP gives posterior mean/variance
`mu_delta(xi), sigma_delta^2(xi)`,

then

`mu(xi) = m_cal(xi) + mu_delta(xi)`,
`sigma^2(xi) = sigma_delta^2(xi)`.

### 8.4 Acquisition

Let `D_f` be the feasible evaluated configurations and

`y_best = max_{xi in D_f} y(xi)`.

Expected improvement:

`EI(xi) = (mu(xi)-y_best) Phi(z_xi) + sigma(xi) varphi(z_xi)`,

where

`z_xi = (mu(xi)-y_best)/sigma(xi)`.

At each outer iteration, generate a finite structurally valid, nonrepeated candidate pool and select

`xi_next = argmax EI(xi)`.

Only `xi_next` invokes the inner solver.

Feasible evaluations update calibration and residual GP. Communication-infeasible or numerical-failure evaluations consume the online evaluation budget, are deduplicated, and must not receive artificial CRB labels.

## 9. Method-status rules

- Do not claim that GP, EI, ridge regression, or Bayesian optimization are newly invented.
- The intended contribution is the problem-specific model adaptation, exact CRB treatment, feasible inner solver, and historical physics-based prior for expensive combinatorial search.
- The online residual GP is not yet established as a standalone contribution relative to the frozen prior; this requires ablation evidence.
- Do not mix historical algorithms/results from incompatible waveform, feature, or solver versions.
