#!/usr/bin/env python3
"""PRL one-column Fig. 3: two SIDE-BY-SIDE panels, no eigensolve.

Usage:
 python plot_fig3.py fig3_spectrum_from_console.csv --fitmax 5
 python plot_fig3.py FIG2_DATA_FOLDER --fitmax 5

CSV columns: l, Re_z, Im_z, gamma (additional columns accepted).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
# Respect MPLBACKEND; run.py selects Agg for portable batch rendering.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

BLUE='#326BA0'
RED='#CE3B33'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path,help='spectrum.csv, console CSV, or data folder')
    ap.add_argument('--fitmax',type=int,default=5,help='Fit positive modes 1 through fitmax')
    ap.add_argument('--out',type=Path,default=None)
    ap.add_argument('--width',type=float,default=3.40)
    args=ap.parse_args()
    inp=args.source/'spectrum.csv' if args.source.is_dir() else args.source
    if not inp.is_file():
        raise FileNotFoundError(f'Spectrum not found: {inp}')
    arr=np.loadtxt(inp,delimiter=',',skiprows=1,ndmin=2)
    if arr.shape[1]<4:
        raise ValueError('Expected l, Re_z, Im_z, gamma as first 4 columns')
    ell=np.rint(arr[:,0]).astype(int)
    zr,zi=arr[:,1],arr[:,2]
    gamma=-zr
    fit=(ell>=1)&(ell<=args.fitmax)
    if fit.sum()<2:
        raise ValueError('Need at least two positive modes in chosen fit range')
    xx=ell[fit].astype(float)**2
    a=np.dot(xx,gamma[fit])/np.dot(xx,xx)
    D=2*a
    print(f'Quadratic fit (1<=l<={args.fitmax}): gamma = {a:.9g} l^2, D_tilde={D:.9g}')
    plt.rcParams.update({
      'font.family':'serif','mathtext.fontset':'cm',
      'font.size':7.1,'axes.labelsize':8.3,
      'xtick.labelsize':6.6,'ytick.labelsize':6.6,
      'legend.fontsize':6.4,'axes.linewidth':.78,
      'xtick.direction':'in','ytick.direction':'in',
      'xtick.top':True,'ytick.right':True,
      'xtick.major.size':2.8,'ytick.major.size':2.8,
      'xtick.major.width':.7,'ytick.major.width':.7,
      'pdf.fonttype':42,'ps.fonttype':42,
    })
    # Both subplots side-by-side; entire canvas fits one journal column.
    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(args.width,args.width*.65))
    fig.subplots_adjust(left=.16,right=.98,bottom=.21,top=.87,wspace=.47)

    fmax=max(abs(-zi))
    omega_est=np.median((-zi[(ell>0)&(ell<=args.fitmax)]/ell[(ell>0)&(ell<=args.fitmax)]))
    if omega_est<=0: raise ValueError("Oscillation frequency must be positive")
    gx=np.linspace(-fmax,fmax,400)
    ax0.plot(gx,a*(gx/omega_est)**2,color=RED,lw=1.0,linestyle=(0,(5,3)),zorder=1)
    ax0.scatter(-zi,gamma,s=10.5,facecolor=BLUE,edgecolor='white',linewidth=.4,zorder=2)
    ax0.set(xlabel=r'$-\operatorname{Im}z_l$',ylabel=r'$-\operatorname{Re}z_l$')
    ax0.set_xlim(-fmax*1.2,fmax*1.2)
    ax0.set_ylim(-.05*max(gamma),max(gamma)*1.09)
    from matplotlib.ticker import MaxNLocator
    print("fmax=", fmax)
    ax0.xaxis.set_major_locator(MaxNLocator(6))

    positive=ell>=0
    ll=ell[positive].astype(float)**2
    yy=gamma[positive]
    j=np.argsort(ll);ll=ll[j];yy=yy[j]
    xfit=np.linspace(0,max(ll),200)
    ax1.plot(xfit,a*xfit,color=RED,lw=1.0,linestyle=(0,(5,3)),zorder=1)
    ax1.scatter(ll,yy,s=10.5,facecolor=BLUE,edgecolor='white',linewidth=.4,zorder=2)
    ax1.set(xlabel=r'$l^2$',ylabel=r'$\gamma_l$')
    ax1.set_xlim(-.03*max(ll),max(ll)*1.1)
    ax1.set_ylim(-.05*max(yy),max(yy)*1.2)
    ax1.xaxis.set_major_locator(MaxNLocator(4))

    # Custom legend proxies; guaranteed small strokes and long samples.
    handles=[Line2D([0],[0],color=BLUE,marker='o',markerfacecolor=BLUE, linestyle='None',
                    markeredgecolor='white',markeredgewidth=.35,markersize=3.1,lw=.7),
             Line2D([0],[0],color=RED,lw=.9,linestyle=(0,(5,3)))]
    ax1.legend(handles,['numerical',r'$\widetilde D l^2/2$'],
               loc='upper left',bbox_to_anchor=(-.03,1.01),frameon=False,
               handlelength=2.8,handletextpad=.36,labelspacing=.25,borderpad=.12)
    for ax, letter in [(ax0,'(a)'),(ax1,'(b)')]:
        ax.text(-.35, 0.95,letter,transform=ax.transAxes,fontsize=9.2,
                ha='left',va='bottom',fontfamily="Times New Roman",clip_on=False)
    out=args.out or ((args.source if args.source.is_dir() else inp.parent)/'Fig3_python_singlecolumn')
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'),dpi=450)
    plt.close(fig)
    print(f'Saved {out.with_suffix(".pdf")} and {out.with_suffix(".png")}')

if __name__=='__main__':
    main()
