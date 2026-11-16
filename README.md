# Limit-cycle figures 1-6: positive slow-sector reference

The reference uses a centered Gaussian distribution of harmonic weights.
The mode metric uses Hilbert–Schmidt-normalized right eigenvectors.

## Install or update

For an existing project, copy `src/`, `plots/`, `tests/`, and `run.py` from this
package into your existing `limit_cycle_figures` folder. Keep your own
`config.toml`, `results/`, `results/cache/`, and Julia environment files.
A missing `[positivity]` section automatically selects the safe defaults below.
Do not overwrite your local plotting edits without keeping a copy.

For a fresh installation, extract the complete folder and run inside it:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
python3 -m pip install -r requirements.txt
python3 run.py
```

The usual command remains `python3 run.py`. Times and physical parameters still
come from `config.toml`. The code reuses an unchanged saved spectral basis;
the first updated build recomputes the affected reference/geometry exports.
Changing positivity settings does not change the basis-cache key.

For a small end-to-end check:

```bash
python3 run.py --config config_smoke.toml --work smoke_results
```

The smoke preset uses N=80, K=5 and kappa=0.02. It is a pipeline check with
physical parameters different from the manuscript. The older N=48/kappa=0.05
preset could fail the existing harmonic-frequency selection criterion at l=5.

For normal run with large size 
```bash
python3 run.py --config config.toml --work results
```

Parameters are: N=450, K=30, and kappa=0.002. It yields all figures used in the paper contained in the folder ./results/figures/

Other commands:

```bash
python3 run.py --data-only
python3 run.py --plot-only --figures 2
python3 run.py --plot-only --force-plots
python3 run.py --dry-run
```

## The physical model

The Hamiltonian and dissipators remain:

```
H = omega0*a†a + epsilon/2*(a²+a†²)
L(rho) = -i[H,rho] + gamma_up*D[a†](rho)
         + gamma_down*D[a](rho) + kappa*D[a²](rho)
```

The production preset retains omega0=1, epsilon=0.6, gamma_up=1,
gamma_down=0.2, kappa=0.002, N=400 and K=50. These numerical settings have not
been certified for spectral or Fock-cutoff convergence. Inspect mode residuals,
top-Fock populations and convergence in N and K. No positivity filter can
correct a wrongly selected Liouvillian mode.

## Gaussian reference and mode metric

Normalize each right eigenvector independently as r_l = rho_l / ||rho_l||_HS.
If R has columns r_l, the code represents eta = R R† implicitly. The
effective orthonormal frame is eta^(-1/2) R on the selected span.
The evolved coherent overlaps at the fitting time set theta=0 for the Gaussian reference.
For a vanishing overlap the phase is fixed by the largest component of the
right mode. Conjugate partners are paired consistently. The projector
r_l r_l† is independent of either choice of phase.

The initial probabilities and their evolution are

```
p_l(0) = exp(-l²/(2*sigma²)) / sum_m exp(-m²/(2*sigma²))
p_l(t) proportional to p_l(0)*exp(2*Re(z_l)*t)
```

Mapping the effective ket back to Liouville space and fixing its trace uses
the physical coefficients c_l = ||rho_s||_HS exp(-l²/(4*sigma²)) in the
phase-aligned frame. This follows because only l=0 has nonzero trace.
The overall scalar used for trace normalization cancels from the effective
probabilities. The same phase coordinate theta enters exp(i*l*theta) in
the state and its derivative.

| Figure | Data |
|---|---|
| 1(a,b) | Classical orbit and stationary Husimi Q |
| 1(c) | Gaussian-initialized density family and its phase derivative |
| 1(d) | Effective phase density from the same Husimi-Q reduction as Fig. 5, with a dotted uniform reference |
| 2(a-d) | Evolved Gaussian weights, fidelity, FS metric and geometric frequency |
| 3 | Slow eigenvalues and diffusion fit |
| 4 | Phase and amplitude comparisons from coherent inputs and left-mode overlaps |
| 5 | Stationary Husimi distribution |
| 6 | Gaussian and evolved-projection harmonic weights at the fitting time, with their residuals |

Fig. 1 is a 2×2 figure with a width of 3.40 inches. To redraw it from a
completed run, use `python3 run.py --plot-only --figures 1` with the
same configuration and work directory used to generate the data.

Fig. 6 uses the saved shifted-time Gaussian data and is included in the
default build. Redraw it with `python3 run.py --plot-only --figures 6`.
In addition to PDF and PNG, this writes `Fig6_harmonics.csv` and
`Fig6_time.csv` in the figures directory. To include Hilbert–Schmidt
errors in the time CSV, call the script with the matching trusted basis:

```bash
python3 plots/fig6.py results/data --basis results/cache/basis_YOUR_HASH.jls --out results/figures/Fig6
```

## Automatic positivity controls

```toml
[positivity]
scope = "orbit"
# match_time = 5.0 # Default is the first positive observation time.
# sigma = 1.0
psd_tolerance = 1e-10
trace_tolerance = 1e-8
precision_accuracy = 1e-6
search_steps = 18
theta_grid = 32
theta_grid_max = 256
certificate_margin = 1e-9
```

The coherent seed is taken from the sample with the largest x²+p² in one
settled mean-field cycle (cycle.csv). The program records its cycle time
t_star and coordinates x0,p0 in positivity_report.txt.

When sigma is omitted, evolve the projected seed to tau = match_time. Fit
a fresh centered Gaussian to those weights at the new time origin s=0,
where s=t-tau:

```
q_l(tau) proportional to ||P_l rho_0||_HS² * exp(2*Re(z_l)*tau)
p_l(s=0) proportional to exp(-l²/(2*sigma²))
Var_p(s=0)(l) = Var_q(tau)(l)
```

Only the projection is evolved during the matching step. The default tau
is the earliest positive observation time, including the Bures observation.
Set match_time = 0 to start both at the same time. For l=-K,...,K the fresh
Gaussian variance has a ceiling of K(K+1)/3 at infinite width. If the target
exceeds that ceiling, the uniform trial is used and the mismatch is reported.
Observation and Bures times, and the time labels in Figs. 1(c) and 2,
are elapsed times s after Gaussian initialization. For example, with
match_time = 4.0, the plotted Gaussian at s=2 corresponds to t=6 after
the coherent projection.

A supplied sigma sets the trial width directly. The program narrows the
Gaussian if necessary until the positivity certificate passes, then refines
the width bracket. The selected width is limited by the numerical certificate
and the requested starting width. It can be smaller than the widest
mathematically admissible Gaussian.

With scope="orbit", certification covers all angles at t=0 and the saved
observation times relative to the Gaussian start, including the Bures time. The scope="reference" setting
certifies theta=0 at those times. Bures samples each undergo their own
PSD, trace and Hermiticity checks. The former lambda settings are accepted
for compatibility and ignored with a warning.

### How the whole-angle certificate works

Write the stationary component as rho_s and the positive-harmonic projected
components as V_l. The Hermitian family is

```
rho(theta,t) = rho_s + sum_{l>0} [a_l(t) V_l exp(i*l*theta) + h.c.]
a_l(t) = exp(-l²/(4*sigma²) + z_l*t).
```

For the proof only, form S=(rho_s+tol*I)^(-1/2), represented in the eigenbasis
of rho_s. Positivity of `I + sum[a_l S† V_l S exp(i*l*theta)+h.c.]` guarantees
`rho(theta,t) >= -tol*I`. The tolerance shift is a numerical certification device;
it is NEVER added to the density matrix.

A direct operator-norm dominance bound is tried first. If it is insufficient,
the code evaluates the Hermitian matrix on an angle grid and bounds its
interpolation error by

```
error <= (h²/8) * 2*sum_l l²*|a_l|*||S† V_l S||_F,   h=2*pi/M.
```

The Frobenius norm bounds the operator norm. Subtracting this error from the
minimum sampled eigenvalue certifies the entire continuous phase circle.
Grid refinement tightens the bound. Ordinary Float64 arithmetic is used; this
is a numerical-tolerance certificate, not exact interval arithmetic.

The program stops if the stationary input fails the numerical PSD test or
if no Gaussian width can be certified within the search range. The tolerance
sets the bound rho >= -psd_tolerance*I. A rank-deficient stationary state
can limit the attainable width.

## Saved diagnostics

- `positivity_report.txt` gives the trial and selected sigma, Gaussian phase
  variance, numerical tolerances, certified times and reference eigenvalue.
  It includes the projected, trial Gaussian and selected Gaussian variances at
  match_time. It also reports half the sum of absolute weight differences at
  the seed time and at match_time, comparing both with the fresh Gaussian.
  When the selected width reaches the target, its initial FS metric agrees
  with the projected variance at tau. Differences in the full distributions may persist.
- `reference_state_real.csv` and `reference_state_imag.csv` give the density
  matrix at theta=0, t=0.
- `reference_harmonics.csv` gives normalized Gaussian weights and the original
  and Gaussian physical coefficients.
- `bures_min_eigenvalue.csv`, `bures_trace_error.csv` and
  `bures_hermiticity_error.csv` describe the plotted density matrices.

An exact complete nondecaying-sector projection is positive by the
stroboscopic CPTP argument in the manuscript. Here the Gaussian is a
parametrized approximation on the retained finite-size sector. The program
checks the reconstructed matrices because Gaussian probabilities by themselves
do not guarantee positivity after mapping back to Liouville space.

## Verification

Tested with Julia 1.10.10 and Arpack 0.5.4:

```bash
julia --project=. tests/test_positivity.jl
julia --project=. tests/test_bures.jl
python3 -m unittest discover -s tests -v
```

Tests cover Gaussian weights, invariance under mode rescaling and rephasing,
trace and Hermiticity, phase derivatives, and an angular minimum hidden
between grid points. The smoke preset tests the full figure workflow.
