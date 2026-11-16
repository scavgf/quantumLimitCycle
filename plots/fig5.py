#!/usr/bin/env python3
"""Reduce Husimi Q(x,p) to a normalized effective-phase density p(theta).

Usage:
  python3 plot_reduced_phase.py results --out Phase_reduced
  python3 plot_reduced_phase.py results --Delta 1 --epsilon 0.6 --bins 120 --out Phase_reduced

By default both P(theta) and P(phi) are plotted; use --theta-only to omit P(phi).

Input: results/husimi_q.csv with three columns x,p,Q(x,p), as exported by
calculate_parametric_vdp.jl. Uses effective theta=arg(sqrt(Delta+epsilon)x
 + i sqrt(Delta-epsilon)p). Valid for the non-Kerr squeezed model with
Delta > abs(epsilon). All counts are integrated with dx dp (the common
factor 1/2 mapping to d^2alpha cancels on normalization).
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
# Respect MPLBACKEND; run.py selects Agg for portable batch rendering.
import matplotlib.pyplot as plt


def read_params(folder, Delta, epsilon):
    meta = folder / 'metadata.txt'
    params = {}
    if meta.exists():
        for line in meta.read_text().splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                try:
                    params[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    D = Delta if Delta is not None else params.get('Delta')
    e = epsilon if epsilon is not None else params.get('epsilon')
    if D is None or e is None:
        raise ValueError('Delta and epsilon not found in metadata.txt; pass --Delta and --epsilon')
    if D <= abs(e):
        raise ValueError('This analytic effective-phase map requires Delta > |epsilon|.')
    return D, e


def periodic_smooth(a, sigma_bins):
    if sigma_bins <= 0:
        return a
    radius = max(1, int(np.ceil(4 * sigma_bins)))
    m = np.arange(-radius, radius+1)
    kernel = np.exp(-0.5 * (m / sigma_bins)**2)
    kernel /= kernel.sum()
    out = sum(kernel[j] * np.roll(a, int(m[j])) for j in range(len(m)))
    return out


def phase_distributions(folder, D, e, bins=120, smooth=1.0):
    """Return bin centers, effective and physical phase densities, and raw Q mass."""
    if bins < 8:
        raise ValueError('--bins must be >= 8')
    data = np.loadtxt(folder / 'husimi_q.csv', delimiter=',', ndmin=2)
    if data.shape[1] < 3:
        raise ValueError('husimi_q.csv must contain x,p,Q columns')
    x, p, Q = data[:,0], data[:,1], data[:,2]
    xs, ps = np.unique(x), np.unique(p)
    if len(xs) < 2 or len(ps) < 2:
        raise ValueError('At least two x-grid and p-grid values are needed')
    dx = np.median(np.diff(xs)); dp = np.median(np.diff(ps))
    if not np.allclose(np.diff(xs), dx) or not np.allclose(np.diff(ps), dp):
        raise ValueError('Input must be a uniform Cartesian Q grid')
    Q = np.maximum(Q, 0)
    w = Q * dx * dp / 2.0   # physically correct phase-space measure for alpha=(x+ip)/sqrt(2)
    raw_total = w.sum()
    if raw_total <= 0:
        raise ValueError('No positive Q weight')
    # On a complete infinite grid: integral Q d²alpha = 1.
    theta = np.mod(np.arctan2(np.sqrt(D-e)*p, np.sqrt(D+e)*x), 2*np.pi)
    physical_phi = np.mod(np.arctan2(p, x), 2*np.pi)
    edges = np.linspace(0, 2*np.pi, bins+1)
    binwidth = edges[1] - edges[0]
    centers = (edges[:-1]+edges[1:])/2
    hist = np.histogram(theta, bins=edges, weights=w)[0]
    density = periodic_smooth(hist / (raw_total * binwidth), smooth)
    density /= np.sum(density) * binwidth
    polar = np.histogram(physical_phi, bins=edges, weights=w)[0]
    polar = periodic_smooth(polar/(raw_total*binwidth), smooth)
    polar /= np.sum(polar)*binwidth
    return centers, density, polar, raw_total


def run():
    plt.rcParams.update({
        'font.family':'Times New Roman',
        'mathtext.fontset': 'stix',
        })
                        
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results', type=Path, help='Folder containing husimi_q.csv')
    ap.add_argument('--out', default='Phase_reduced', help='Output path stem')
    ap.add_argument('--Delta', type=float, default=None)
    ap.add_argument('--epsilon', type=float, default=None)
    ap.add_argument('--bins', type=int, default=120)
    ap.add_argument('--smooth', type=float, default=1.0,
                    help='Periodic Gaussian smoothing width in histogram bins (default 1)')
    ap.add_argument('--theta-only', action='store_true',
                    help='Show only effective p(theta); default compares p(theta) and p(phi)')
    args = ap.parse_args()
    D, e = read_params(args.results, args.Delta, args.epsilon)
    centers, density, polar, raw_total = phase_distributions(args.results, D, e, args.bins, args.smooth)
    binwidth = 2*np.pi / args.bins
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out.with_name(out.name + '_p_theta.csv'),
               np.column_stack([centers/np.pi, density]), delimiter=',',
               header='theta_over_pi,p_theta_per_radian', comments='')
    fig, ax = plt.subplots(figsize=(6.3, 3.65), constrained_layout=True)
    tplot = np.r_[centers, centers[0]+2*np.pi]
    dplot = np.r_[density, density[0]]
    ax.plot(tplot/np.pi, dplot, lw=2, label=r'$P(\theta)$', color='#236da8')
    np.savetxt(out.with_name(out.name + '_phase_comparison.csv'),
               np.column_stack([centers/np.pi, density, polar]), delimiter=',',
               header='phase_over_pi,p_theta_per_radian,p_phi_per_radian', comments='')
    if not args.theta_only:
        ax.plot(tplot/np.pi, np.r_[polar, polar[0]], lw=1.5, ls='--',
                label=r'$P(\phi)$', color='#c46f32')
    ax.legend(frameon=False, fontsize=18)
    ax.axhline(1/(2*np.pi), color='#777777', lw=1, ls=':', label=None)
    ax.set_xlabel(r'phase $/\pi$', fontsize=18)
    ax.set_ylabel(r'phase distribution', fontsize=18)
    ax.set_xticks([0,.5,1,1.5,2])
    ax.tick_params(direction='in', top=True, right=True, labelsize=18)
    ax.set_ylim(bottom=0)
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=240)
    plt.close(fig)
    print(f'Delta={D:g}, epsilon={e:g}; ∫grid Q d²alpha = {raw_total:.8g}')
    print(f'Normalized ∫ P(theta) dtheta = {np.sum(density)*binwidth:.12g}')
    print(f'Normalized ∫ P(phi) dphi = {np.sum(polar)*binwidth:.12g}')
    print('Saved:', out.with_suffix('.png'), out.with_suffix('.pdf'))

if __name__ == '__main__':
    run()
