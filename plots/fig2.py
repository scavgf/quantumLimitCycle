#!/usr/bin/env python3
"""Four-panel phase geometry figure (3.40 inch single-column width).

Run from the organized project's root:
  python3 plots/fig2.py results/data --out results/figures/Fig2_2x2

Reads weights.csv, fidelity.csv, metric_time.csv, fs_selected.csv and
spectrum.csv (or legacy mode_spectrum.csv). No eigensolves or model changes.

QGT: Q_mu,nu=<partial_mu psi|(1-|psi><psi|)|partial_nu psi>,
g_mu,nu=Re Q_mu,nu. For psi_l proportional to c_l exp(z_l*t) exp(i*l*theta),
z_l=-gamma_l-i*omega_l:
  g_theta,theta=Var(l), g_theta,t=-Cov(l,omega_l).
The signed frequency is g_theta,t/g_theta,theta. Panel (d) divides this by
--omega (omega0) for dimensionless axes; the exported CSV retains raw values.
Changing the theta orientation reverses the mixed-metric sign.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # works on macOS, Linux, and headless clusters
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

COLORS = ['#2C6B9C', '#CE6F33', '#248E78', '#9166AA', '#B99B2E',
          '#B64B53', '#5797AF', '#7D7972']


def styling():
    plt.rcParams.update({
        'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
        'font.size': 7.7, 'axes.labelsize': 9.0,
        'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2,
        'legend.fontsize': 6.7, 'axes.linewidth': .75,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
        'xtick.major.width': .75, 'ytick.major.width': .75,
        'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white',
    })


def load(folder, name):
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError(f'Required CSV not found: {path}')
    arr = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f'Empty or nonfinite data in {path}')
    return arr


def diffusion_prediction(t, ell, p0, D):
    """Exact discrete-weight variance under gamma_l = D*l**2/2."""
    if D < 0:
        raise ValueError('Diffusion rate must be nonnegative')
    t = np.asarray(t)
    p0 = np.asarray(p0, dtype=float)
    if np.any(p0 < 0) or np.sum(p0) <= 0:
        raise ValueError('Initial spectral weights must be nonnegative')
    p0 = p0 / p0.sum()
    # Stable for large time or high harmonic cutoff.
    logw = np.log(np.maximum(p0, 1e-300))[None, :] - D*t[:, None]*ell[None, :]**2
    logw -= logw.max(axis=1, keepdims=True)
    weights = np.exp(logw)
    weights /= weights.sum(axis=1, keepdims=True)
    mean = weights @ ell
    variance = weights @ (ell**2) - mean**2
    return np.divide(1.0, variance, out=np.full_like(variance,np.nan), where=variance > 0)



def geometric_frequency(time_grid, ell, p_ref, t_ref, spectrum):
    """Compute g_{theta,t}/g_{theta,theta} from the SAME slow eigenmodes.

    Conventions: z_l=-gamma_l-i*omega_l,
    g_{theta,t}=-Cov_t(l,omega_l); g_{theta,theta}=Var_t(l).
    The input p_ref is the normalized harmonic distribution at t_ref.
    """
    if spectrum.shape[1] < 3:
        raise ValueError('mode_spectrum.csv must have columns l,Re_z,Im_z')
    labels = spectrum[:, 0]
    if len(np.unique(labels)) != len(labels):
        raise ValueError('Duplicate l values in mode_spectrum.csv')
    order = np.argsort(labels)
    labels, spec = labels[order], spectrum[order]
    if len(labels) != len(ell) or not np.allclose(labels, ell, rtol=0, atol=1e-8):
        raise ValueError('Harmonics in mode_spectrum.csv do not match weights.csv')
    gamma = -spec[:, 1]
    omega_l = -spec[:, 2]
    if np.any(gamma < -1e-8):
        raise ValueError('Positive real part of some selected Liouvillian modes')
    if np.any(np.asarray(time_grid) < t_ref - 1e-8):
        raise ValueError('Time grid must start at or after reference time')
    p_ref = np.array(p_ref, dtype=float, copy=True)
    if np.any(p_ref < 0) or p_ref.sum() <= 0:
        raise ValueError('Invalid reference harmonic weights')
    p_ref /= p_ref.sum()
    logp = np.full(p_ref.shape, -np.inf)
    positive = p_ref > 0
    logp[positive] = np.log(p_ref[positive])
    logw = logp[None, :] - 2.0*(np.asarray(time_grid)[:, None]-t_ref)*gamma[None, :]
    logw -= np.max(logw, axis=1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(axis=1, keepdims=True)
    mean_l = w @ ell
    mean_omega = w @ omega_l
    centered_l = ell[None, :] - mean_l[:, None]
    centered_omega = omega_l[None, :] - mean_omega[:, None]
    variance = np.sum(w*centered_l**2, axis=1)
    covariance = np.sum(w*centered_l*centered_omega, axis=1)
    ratio = np.divide(-covariance, variance, out=np.full_like(variance, np.nan),
                      where=variance > max(1e-12, 1e-10*variance[0]))
    return ratio, variance

def add_label(ax, label, x=-.12):
    ax.text(x, 0.955, label, transform=ax.transAxes, fontfamily="Times New Roman",
            fontsize=10.5, ha='left', va='bottom', clip_on=False)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder', type=Path, help='Directory with Julia output CSVs')
    p.add_argument('--out', type=Path, default=Path('Fig2_2x2'),
                   help='Output filename stem (default: Fig2_final)')
    p.add_argument('--D', type=float, default=None,
                   help='Diffusion coefficient; default: fit saved harmonics 1..5; 0 omits theory')
    p.add_argument('--width', type=float, default=3.40,
                   help='Figure width in inches (default: 3.40)')
    p.add_argument('--height', type=float, default=3.55,
                   help='Figure height in inches (default: 3.55)')
    p.add_argument('--times', type=str, default=None,
                   help='Comma-separated saved times for panels (a),(b); default: all saved')
    p.add_argument('--omega', type=float, default=1.0,
                   help='Omega_0: scales saved t for axis and time legends (default: 1)')
    p.add_argument('--dpi', type=int, default=600,
                   help='PNG resolution (default: 600)')
    args = p.parse_args()
    if not np.isfinite(args.omega) or args.omega <= 0:
        p.error('--omega must be finite and positive')
    if args.width <= 0 or args.height <= 0 or args.dpi <= 0:
        p.error('Figure dimensions and DPI must be positive')
    styling()

    weights = load(args.folder, 'weights.csv')
    fidelity = load(args.folder, 'fidelity.csv')
    metric = load(args.folder, 'metric_time.csv')
    fs = load(args.folder, 'fs_selected.csv')
    spec_name = 'spectrum.csv' if (args.folder/'spectrum.csv').is_file() else 'mode_spectrum.csv'
    spectrum = load(args.folder, spec_name)
    if args.D is None:
        mask = (spectrum[:, 0] >= 1) & (spectrum[:, 0] <= 5)
        xx = spectrum[mask, 0]**2
        if len(xx) < 2:
            p.error('Need at least two positive modes for the diffusion fit; use --D 0 to omit')
        args.D = 2*np.dot(xx, -spectrum[mask, 1])/np.dot(xx, xx)
    if not np.isfinite(args.D) or args.D < 0:
        p.error('--D must be finite and nonnegative')
    if metric.shape[1] < 2 or fs.shape[1] < 2:
        raise ValueError('metric_time.csv and fs_selected.csv need columns t,g_FS')
    times = fs[:, 0]
    ell = weights[:, 0]
    delta = fidelity[:, 0]
    n = len(times)
    if weights.shape[1] != n + 1 or fidelity.shape[1] != n + 1:
        raise ValueError(f'Expected {n} time columns in weights/fidelity CSVs')
    if len(times) != len(np.unique(times)):
        raise ValueError('Duplicate selected times in fs_selected.csv')

    if args.times is None:
        selected = list(range(n))
    else:
        selected = []
        for s in args.times.split(','):
            if not s.strip():
                continue
            tval = float(s)
            ix = np.flatnonzero(np.isclose(times, tval, rtol=1e-6, atol=1e-9))
            if len(ix) != 1:
                raise ValueError(f'Time {tval} not present; saved times: {times.tolist()}')
            if int(ix[0]) not in selected:
                selected.append(int(ix[0]))
    if not selected:
        raise ValueError('No plotting times selected')

    # PRL one-column width: exactly two columns and two equal rows.
    fig = plt.figure(figsize=(args.width, args.height))
    gs = fig.add_gridspec(2, 2, left=.13, right=.95,
                          bottom=.105, top=.99,
                          wspace=.53, hspace=.33)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel (a): weights. Shared color/time mapping for top row.
    for j, i in enumerate(selected):
        color = COLORS[j % len(COLORS)]
        ax_a.plot(ell, weights[:, i+1], color=color, lw=1.15,
                  marker='o', ms=1.7, mew=0, markevery=max(1,len(ell)//32),
                  label=rf'${args.omega*times[i]:g}$')
    ax_a.set_xlabel(r'$l$', labelpad=2.4)
    ax_a.set_ylabel(r'$p_l(t)$', labelpad=3)
    ax_a.set_xlim(float(ell.min()), float(ell.max()))
    topmax = float(np.max(weights[:, np.array(selected)+1]))
    ax_a.set_ylim(0, 1.10*topmax if topmax > 0 else 1)
    ax_a.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax_a.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_a.legend(title=r'$\omega_0 t$', loc='upper right',
                fontsize=6.05, title_fontsize=6.3,
                frameon=False, ncol=1, handlelength=1.25,
                borderaxespad=.25, labelspacing=.11,
                handletextpad=.35)

    # Panel (b): fidelity, using same colors and order.
    for j, i in enumerate(selected):
        ax_b.plot(delta, fidelity[:, i+1], color=COLORS[j % len(COLORS)], lw=1.3)
    ax_b.set_xlabel(r'$\Delta\theta/\pi$', labelpad=2.4)
    ax_b.set_ylabel(r'$\mathcal{F}_t(\Delta\theta)$', labelpad=3)
    ax_b.set_xlim(float(delta.min()), float(delta.max()))
    ax_b.set_ylim(0, 1.055)
    ax_b.xaxis.set_major_locator(MultipleLocator(.5))
    ax_b.yaxis.set_major_locator(MultipleLocator(.25))

    # Panel (c): inverse FS metric and full discrete diffusion prediction.
    time_grid, metric_fs = metric[:, 0], metric[:, 1]
    # Start at the earliest saved weight reference, including t=0 if present.
    t_ref = float(np.min(times))
    valid = (time_grid >= t_ref - 1e-9) & (metric_fs > 0) & np.isfinite(metric_fs)
    time_grid, metric_fs = time_grid[valid], metric_fs[valid]
    if len(time_grid) < 2:
        raise ValueError('Not enough positive FS metric samples in metric_time.csv')
    x = args.omega * time_grid
    num = 1.0 / metric_fs
    ax_c.plot(x, num, color='#2C6B9C', lw=1.4,
              label='Liouvillian')
    predicted = None
    if args.D > 0:
        idx_ref = int(np.argmin(times))
        # Initial condition is p_l(t_ref), not necessarily p_l(0).
        # Diffusion acts over elapsed time; no backward extrapolation.
        predicted = diffusion_prediction(time_grid - t_ref, ell, weights[:, idx_ref+1], args.D)
        ax_c.plot(x, predicted, color='#CF6E2B', lw=1.2,
                  ls=(0,(5.0,2.5)), label='Phase diffusion')
    ax_c.set_xlabel(r'$\omega_0 t$', labelpad=2.4)
    ax_c.set_ylabel(r'$g_{\mathrm{FS}}^{-1}(t)$', labelpad=3)
    ax_c.set_xlim(float(x.min()), float(x.max()))
    all_y = np.concatenate((num, predicted[np.isfinite(predicted)])) if predicted is not None else num
    all_y = all_y[np.isfinite(all_y)]
    ymin, ymax = float(all_y.min()), float(all_y.max())
    span = max(ymax-ymin, 0.08*max(abs(ymax),1))
    ax_c.set_ylim(max(0,ymin-.12*span), ymax+.16*span)
    ax_c.xaxis.set_major_locator(MaxNLocator(4))
    ax_c.yaxis.set_major_locator(MaxNLocator(5))
    ax_c.legend(loc='lower right', frameon=False, ncol=1,
                handlelength=1.4, labelspacing=.15,
                fontsize=5.7, borderaxespad=.22)

    # Panel (d): geometric frequency. All values from *the same* Liouvillian
    # eigenmodes as the weight evolution; no harmonic-spacing assumption.
    omega_geo, g_from_modes = geometric_frequency(
        time_grid, ell, weights[:, int(np.argmin(times))+1].copy(), t_ref, spectrum)
    if not np.allclose(g_from_modes, metric_fs, rtol=1e-5, atol=1e-9):
        raise ValueError('Spectrum/weights do not reproduce metric_time.csv; do not mix data from different calculations')
    valid_geo = np.isfinite(omega_geo)
    if not np.any(valid_geo):
        raise ValueError('Cannot calculate geometric frequency: zero FS variance')
    ax_d.plot(args.omega*time_grid[valid_geo],
              -omega_geo[valid_geo]/args.omega, color='#248E78', lw=1.45,
              label='Liouvillian')

    ax_d.set_xlabel(r'$\omega_0 t$', labelpad=2.4)
    ax_d.set_ylabel(r'$\omega_{g}(t)$', labelpad=1.8)
    ax_d.set_xlim(float(x.min()), float(x.max()))
    ax_d.set_ylim(0.7,0.9)
    ax_d.xaxis.set_major_locator(MaxNLocator(4))
    ax_d.yaxis.set_major_locator(MaxNLocator(4))


    # Panel marks, placed inside axes to minimize margins at single-column size.
    for ax, tag in [(ax_a, '(a)'), (ax_b, '(b)'), (ax_c, '(c)'), (ax_d, '(d)')]:
        ax.text(.03, .97, tag, transform=ax.transAxes, ha='left', va='top',
                fontsize=9.5)

    out = args.out
    if out.suffix.lower() in ('.png','.pdf'):
        out = out.with_suffix('')
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_name(out.name + '_geometry_frequency.csv')
    np.savetxt(csv_path, np.column_stack((time_grid, args.omega*time_grid,
               g_from_modes, omega_geo*g_from_modes, omega_geo, omega_geo/args.omega)),
               delimiter=',', comments='',
               header='t,omega0_t,g_theta_theta,g_theta_t,geometry_frequency,geometry_frequency_over_omega0')
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=args.dpi)
    plt.close(fig)
    print(f'Created: {csv_path}')
    print(f'Created: {out.with_suffix(".pdf")}')
    print(f'Created: {out.with_suffix(".png")}')
    if args.D > 0:
        print(f'Diffusion comparison starts at t_ref = {t_ref:g} with D_tilde = {args.D:.6g}')
    print(f'Geometric frequency: min={np.nanmin(omega_geo):.6g}, max={np.nanmax(omega_geo):.6g}')

if __name__ == '__main__':
    main()
