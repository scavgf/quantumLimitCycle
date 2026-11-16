# Original-file audit

Only active programs are included. No original archive was overwritten.

| Original program | Disposition and reason |
|---|---|
| `parametric_vdp_core.jl` | Retained as the basis of `src/model.jl`; its CLI removed; left eigenvectors retained for reuse |
| `calculate_parametric_vdp.jl` | Removed older duplicate of the core, with an older high-harmonic selector |
| `compare_theta.jl` | Phase-export routines moved to `src/phase.jl`; duplicate Liouvillian and eigensolve removed |
| `fig2_calculate_any_times_geomfreq.jl` | Removed from this manuscript workflow: uses H=omega*n and (kappa/2)D[a*x], unlike Eq. (1)'s parametric H and kappa*D[a²] |
| `plot_fig1.py` | Retained as `plots/fig1.py`; common data folder, portable batch backend |
| `plot_fig2.py` | Retained as `plots/fig2.py`; matches the PDF's three panels; diffusion fit defaults to the shared saved spectrum |
| `plot_fig3.py` | Retained as `plots/fig3.py`; reads the same `spectrum.csv` |
| `plot_theta_comparison.py` | Consolidated as `plots/fig4.py`; two panels matching the caption, physical-angle x axis in panel (b), consistent clockwise orientation |
| `plot_phase_correspondence.py` | Removed alternative phase layout; the retained Fig. 4 program covers the PDF comparison |
| `plot_reduced_phase.py` | Retained as `plots/fig5.py`; uses Fig. 1's Husimi export |
| `plot_fig2_2x2_geomfreq.py` | Removed optional four-panel alternative; the extra geometric-frequency panel is absent from this PDF |
| `__MACOSX/`, `._*` | Removed macOS metadata; no numerical or plotting content |

New infrastructure: `src/calculate.jl`, `run.py`, parameter presets,
Julia/Python dependency files, workflow tests and this documentation.

The retained science routines preserve the original core's Hilbert-Schmidt
normalization, biorthogonal left/right pairing, adaptive quadratic shift for
high harmonics, the spectrum and the classical dynamics.
The production configuration now follows the PDF caption rather than the old
core's epsilon=0.3 and kappa=0.0125 defaults.

## Positive-reference update

The four-panel geometry-frequency program is now the active `plots/fig2.py`.
`src/positivity.jl` constructs the Gaussian initial weights; `src/positive_geometry.jl`
replaces the old surrogate-export path. `src/model.jl` is unchanged for
basis-cache compatibility. The old geometry helpers in that file are not
invoked by `src/calculate.jl`. See README.md for the numerical certificate.
