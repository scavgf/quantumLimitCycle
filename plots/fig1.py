#!/usr/bin/env python3
"""Create a single-column 2×2 Fig. 1.

(a) Parametric van der Pol mean-field limit-cycle paths, colored by orbital occupancy.
(b) Stationary Husimi-Q from the microscopic Lindblad Liouvillian.
(c) Polar metric comparison based on src/calculate.jl output.
(d) Effective phase density from the same Husimi-Q reduction as Fig. 5.

Requires actual Julia CSV inputs. Bures values use the Gaussian-initialized reference family and its derivative.

Run:
  python plots/fig1.py --classical Fig1_data \\
      --geometry Fig2_SingleRef_N400_K50_M90_T0_1_2_3_4_5 --time 2

Output: Fig1_PRL.pdf and Fig1_PRL.png (default).

Positivity is certified before export; no pointwise eigenvalue clipping is used.
"""
import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import matplotlib
# Respect MPLBACKEND; run.py selects Agg for portable batch rendering.
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

from fig5 import phase_distributions, read_params

BLUE = '#236F9F'
ORANGE = '#D77632'
GRAY = '#898989'


def select_column(headers, t):
    for j, h in enumerate(headers):
        if h.strip().startswith('t_'):
            try:
                if np.isclose(float(h.strip()[2:]), t, atol=1e-8):
                    return j
            except ValueError:
                pass
    raise ValueError(f'No t={t:g} column in {headers}')


def load_wide(path, t):
    with open(path, newline='', encoding='utf-8') as f:
        hdr = next(csv.reader(f))
    data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    j = select_column(hdr, t)
    return data[:,0], data[:,j]


def load_geometry(folder, t):
    fs = np.loadtxt(folder/'fs_selected.csv', delimiter=',', skiprows=1, ndmin=2)
    idx = np.flatnonzero(np.isclose(fs[:,0],t,atol=1e-8))
    if not len(idx):
        raise ValueError(f't={t} not found in fs_selected.csv')
    gfs = float(fs[idx[0],1])
    theta_over_pi, b = load_wide(folder/'bures_absolute.csv',t)
    theta = np.pi*theta_over_pi  # IMPORTANT: source CSV is theta/pi
    flags = folder/'bures_repaired_mask.csv'
    if flags.exists():
        tf, used = load_wide(flags,t)
        if not np.allclose(tf,theta_over_pi):
            warnings.warn('Bures-repair-mask angular grid differs')
        if np.any(used > 0.5):
            warnings.warn(f'{int(np.count_nonzero(used>0.5))}/{len(used)} Bures samples at t={t:g} '
                          'were PSD repaired; these are SURROGATE diagnostics, not genuine Bures metrics.')
    return theta, b, gfs


def set_fonts():
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = 'Times New Roman' if 'Times New Roman' in available else 'STIXGeneral'
    if family != 'Times New Roman':
        warnings.warn('Times New Roman is unavailable; using STIXGeneral (Times-like).')
    plt.rcParams.update({
        'font.family': family,
        'mathtext.fontset': 'stix',
        'font.size': 7.5,
        'axes.linewidth': .6,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 600,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def panel_label(ax, name, *, dx=-0.14, dy=0.045):
    # Axes coordinates keep panel labels visually aligned with their own panels.
    ax.text(dx, 1.0+dy, name, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=9, color='black', clip_on=False)


def draw_classical(ax, folder, extent=26.):
    approach = np.loadtxt(folder/'approach.csv', delimiter=',', ndmin=2)
    for k in np.unique(approach[:,0]):
        rows = approach[approach[:,0]==k]
        ax.plot(rows[:,2], rows[:,3], color='#A6A6A6', lw=.64, alpha=.80, zorder=1)
        # EXACTLY ONE arrow per gray trajectory, near its visible beginning.
        # Use arc length rather than point indices, so the arrow is legible
        # regardless of trajectory sampling density or integration duration.
        points = rows[:, 2:4]
        inside = (np.abs(points[:, 0]) < extent - 1.5) & (np.abs(points[:, 1]) < extent - 1.5)
        starts = np.flatnonzero(inside)
        if len(starts) > 1:
            first = int(starts[0])
            # Follow the first continuous visible portion of this trajectory.
            last = first
            while last + 1 < len(points) and inside[last + 1]:
                last += 1
            visible = points[first:last + 1]
            if len(visible) >= 3:
                ds = np.linalg.norm(np.diff(visible, axis=0), axis=1)
                arc = np.r_[0., np.cumsum(ds)]
                if arc[-1] > 0.5:
                    # Arrow begins at 8% of the visible curve and spans 1.7 data units.
                    target = 0.02 * arc[-1]
                    i0 = min(int(np.searchsorted(arc, target)), len(visible) - 2)
                    i1 = min(int(np.searchsorted(arc, arc[i0] + 1.7)), len(visible) - 1)
                    if i1 > i0:
                        ax.annotate('', xy=visible[i1], xytext=visible[i0],
                                    arrowprops=dict(arrowstyle='-|>', color='#858585',
                                                    lw=0.85, mutation_scale=10,
                                                    shrinkA=0, shrinkB=0),
                                    zorder=3)
    orbit=np.loadtxt(folder/'cycle.csv',delimiter=',',ndmin=2)
    xy=orbit[:,1:3]
    speed=1./np.maximum(orbit[:,3],1e-20)
    # Real weight controls color, not artificial angular shading.
    val = np.log(np.maximum(orbit[:,3],1e-15))
    color_map=LinearSegmentedColormap.from_list('orbitwarm', ['#eeab40','#df7e30','#b9252d'])
    segs=np.stack([xy[:-1],xy[1:]],axis=1)
    lc=LineCollection(segs,cmap=color_map,norm=Normalize(vmin=np.quantile(val,.04),vmax=np.quantile(val,.96)),
                      linewidths=2.7,zorder=4,capstyle='round')
    lc.set_array((val[:-1]+val[1:])/2)
    ax.add_collection(lc)
    ax.set(xlim=(-extent,extent),ylim=(-extent,extent),xlabel=r'$x(t)$',ylabel=r'$p(t)$')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-20,0,20]); ax.set_yticks([-20,0,20])
    ax.tick_params(direction='in',top=True,right=True,length=2.0,width=.5,pad=2)
    ax.grid(False)
    for sp in ax.spines.values(): sp.set_color('#999999'); sp.set_linewidth(.5)


def draw_husimi(ax, folder, extent=26.):
    rows=np.loadtxt(folder/'husimi_q.csv',delimiter=',',ndmin=2)
    x=np.unique(rows[:,0]);p=np.unique(rows[:,1]); n=len(x);m=len(p)
    if len(rows)!=n*m:
        raise ValueError('Incomplete Husimi grid')
    Q=rows[:,2].reshape((m,n)) # ordered by p outer, x inner
    # Gentle off-white/amber/red Husimi palette closely aligned with paper.
    cmap=LinearSegmentedColormap.from_list('qpaper',
        ['#fdfdf2','#fff8bf','#f9d951','#ec9a31','#a92f39'])
    ax.imshow(Q,origin='lower',extent=(x.min(),x.max(),p.min(),p.max()),
              cmap=cmap,interpolation='bilinear',aspect='equal',vmin=0,
              vmax=float(np.max(Q))*1.06,rasterized=True)
    ax.set(xlim=(-extent,extent),ylim=(-extent,extent),xlabel=r'$x$',ylabel=r'$p$')
    ax.set_xticks([-20,0,20]); ax.set_yticks([-20,0,20])
    ax.tick_params(direction='in',top=True,right=True,length=2.0,width=.5,pad=2)
    for sp in ax.spines.values(): sp.set_color('#999999'); sp.set_linewidth(.5)


def draw_polar(ax, theta, b, gfs, *, t=2.,omega=1., scale=2., corotate=True):
    if not(np.isfinite(gfs) and gfs>0):
        raise ValueError('FS metric must be finite and positive')
    valid=np.isfinite(theta)&np.isfinite(b)&(b>=0)
    if valid.sum()<16: raise ValueError('Too few valid Bures samples')
    th=theta[valid].copy(); gb=b[valid]
    if corotate: th -= omega*t
    th=np.mod(th,2*np.pi)
    order=np.argsort(th);th=th[order];gb=gb[order]
    rr=np.sqrt(scale*gb); rfs=np.sqrt(gfs)
    maxr=1.16*max(rfs,np.max(rr))
    theta_fine=np.linspace(0,2*np.pi,721)
    ax.plot(theta_fine,np.full_like(theta_fine,rfs),color=BLUE,lw=1.65,
            label=r'$r_{\rm FS}=\sqrt{g_{\rm FS}}$',zorder=3)
    # Close only for complete angular coverage; don't bridge large gaps.
    dtheta=np.diff(np.r_[th,th[0]+2*np.pi])
    if dtheta.max() < max(.2, 1.5*float(np.median(dtheta))):
        th_plot=np.r_[th,th[0]+2*np.pi];rr_plot=np.r_[rr,rr[0]]
    else:
        warnings.warn('Missing Bures angles: curve is not artificially closed')
        th_plot=th;rr_plot=rr
    ax.plot(th_plot,rr_plot,color=ORANGE,lw=1.4,ls=(0,(4.5,2.3)),
            label=r'$r_{\rm B}=\sqrt{2 g_{\rm B}}$',zorder=4)
    ax.set_theta_zero_location('E');ax.set_theta_direction(1)
    ax.set_ylim(0,maxr)
    # Eight angular ticks, with four additional pi/4 labels.
    ax.set_xticks(np.arange(0, 2*np.pi, np.pi/4))
    ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                        r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'])
    ax.tick_params(axis='x', pad=-2.5, labelsize=6.8)
    ax.set_yticks(maxr*np.array([.2,.4,.6,.8])); ax.set_yticklabels([])
    ax.grid(color='#D7D7D7', lw=.42, alpha=.55)
    ax.spines['polar'].set_color('#D7D7D7');ax.spines['polar'].set_linewidth(.42)
    return rfs


def draw_phase_density(ax, folder, delta, epsilon):
    centers, density, _, _ = phase_distributions(folder, delta, epsilon)
    # Periodically extend the bin-center samples so both endpoints are drawn.
    angles = np.r_[centers[-1]-2*np.pi, centers, centers[0]+2*np.pi]
    values = np.r_[density[-1], density, density[0]]
    ax.plot(angles, values, color=BLUE, lw=1.05)
    ax.axhline(1/(2*np.pi), color=GRAY, ls=':', lw=.85)
    ax.set(xlim=(0,2*np.pi), ylim=(0, 2*max(float(density.max()),1/(2*np.pi))),
           xlabel=r'$\theta$', ylabel=r'$P(\theta)$')
    ax.set_xticks([0,np.pi,2*np.pi], [r'$0$',r'$\pi$',r'$2\pi$'])
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(direction='in',top=True,right=True,length=2,width=.5,pad=2)
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_color('#999999')
        spine.set_linewidth(.5)


def build(args):
    set_fonts()
    folder=Path(args.classical); geo=Path(args.geometry)
    theta,b,gfs=load_geometry(geo,args.time)
    # The Husimi grid and its model parameters must describe the same run.
    params_folder = folder if (folder/'metadata.txt').exists() else geo
    delta,epsilon=read_params(params_folder,args.omega,args.epsilon)

    fig=plt.figure(figsize=(3.4,3.46),facecolor='white')
    axa=fig.add_axes([.141,.579,.325,.3194])
    axb=fig.add_axes([.637,.579,.325,.3194])
    axc=fig.add_axes([.141,.121,.325,.3194],projection='polar')
    axd=fig.add_axes([.637,.121,.325,.3194])
    cycle=np.loadtxt(folder/'cycle.csv',delimiter=',',ndmin=2)
    extent=max(7.,1.23*float(np.max(np.abs(cycle[:,1:3]))))
    draw_classical(axa,folder,extent=extent)
    draw_husimi(axb,folder,extent=extent)
    for ax in (axa,axb):
        ax.xaxis.set_major_locator(MaxNLocator(3,integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(3,integer=True))
    draw_polar(axc,theta,b,gfs,t=args.time,omega=delta,corotate=not args.no_corotate)
    draw_phase_density(axd,folder,delta,epsilon)
    for ax,name in zip((axa,axb,axc,axd),('(a)','(b)','(c)','(d)')):
        panel_label(ax,name)
    handles, labels=axc.get_legend_handles_labels()
    fig.legend(handles,labels,loc='center',bbox_to_anchor=(.3,.031),ncol=2,
               frameon=False,fontsize=7,handlelength=1.7,
               handletextpad=.4,columnspacing=1.1,borderaxespad=0)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'),facecolor='white',bbox_inches=None)
    fig.savefig(out.with_suffix('.png'),facecolor='white',bbox_inches=None,dpi=args.dpi)
    plt.close(fig)
    print(f'Saved {out.with_suffix(".pdf")} and {out.with_suffix(".png")}')
    print(f't={args.time:g}, g_FS={gfs:.8g}; angular Bures variation = '
          f'{(np.nanmax(b)-np.nanmin(b))/np.nanmean(b):.3%}')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--classical',required=True,help='Data folder from src/calculate.jl')
    ap.add_argument('--geometry',required=True,help='Folder from src/calculate.jl')
    ap.add_argument('--time',type=float,default=2.)
    ap.add_argument('--omega',type=float,default=None,help='Override Delta from metadata')
    ap.add_argument('--epsilon',type=float,default=None,help='Override epsilon from metadata')
    ap.add_argument('--no-corotate',action='store_true')
    ap.add_argument('--dpi',type=int,default=600)
    ap.add_argument('--output',default='Fig1_PRL')
    build(ap.parse_args())

if __name__=='__main__':main()
