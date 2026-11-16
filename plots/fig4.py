#!/usr/bin/env python3
"""Publication-oriented comparison of classical and quantum phase coordinates.

Usage:
 python3 plots/fig4.py results/data --out results/figures/Fig4

Input is theta_comparison.csv exported by compare_theta.jl.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path,nargs='?',default=Path('theta_results'))
    p.add_argument('--out',default='Fig4')
    p.add_argument('--harmonics', type=int, nargs='+', default=None)
    p.add_argument('--dpi',type=int,default=600)
    args=p.parse_args()
    data=np.genfromtxt(args.folder/'theta_comparison.csv',delimiter=',',names=True)
    cols=data.dtype.names
    harmonics=sorted(int(n.split('_l')[-1]) for n in cols if n.startswith('theta_quantum_l'))
    if not harmonics:
        raise ValueError('No quantum harmonic columns in theta_comparison.csv')
    if args.harmonics is not None:
        missing = set(args.harmonics) - set(harmonics)
        if missing:
            raise ValueError(f'Missing phase harmonics: {sorted(missing)}')
        harmonics = args.harmonics
    # Choose clockwise-positive orientation for both classical and quantum phases.
    theta=-np.asarray(data['theta_classical'])
    phi=-(np.asarray(data['phi_unwrapped'])-data['phi_unwrapped'][0])
    plt.rcParams.update({
                        'font.family':'Times New Roman','mathtext.fontset':'stix',
                        'xtick.labelsize':12,
                        'ytick.labelsize':12,
                        'legend.fontsize':10,
                         'font.size':13,
                         'axes.linewidth':0.8,'xtick.direction':'in',
                         'ytick.direction':'in','xtick.top':True,'ytick.right':True})
    fig,(a,b)=plt.subplots(1,2,figsize=(6.8,2.72),dpi=160)
    # Left: direct phase comparison with equal scaling.
    a.plot(theta/np.pi,theta/np.pi,color='0.2',lw=1.5,ls='--',label='Identity')
    palette=['#2464a5','#c47735','#438b72','#a75284']
    for j,l in enumerate(harmonics):
        arr=-np.asarray(data[f'theta_quantum_l{l}'])
        a.plot(theta/np.pi,arr/np.pi,color=palette[j%len(palette)],
               lw=1.25,alpha=0.90,label=rf'$l={l}$')
    a.set(xlabel=r'$\theta_{\rm cl}/\pi$',ylabel=r'$\theta_{{\rm q},l}/\pi$')
    a.legend(loc='upper left',frameon=False,ncol=1)
    a.set_aspect('equal',adjustable='box')
    a.text(-0.25,1.02,'(a)',transform=a.transAxes,fontsize=14)
    # Right: signed unwrapped discrepancies vs clockwise physical polar angle.
    for j,l in enumerate(harmonics):
        b.plot(phi/np.pi,-10**3*np.asarray(data[f'phase_error_l{l}']),
               color=palette[j%len(palette)],lw=1.35,label=rf'$l={l}$')
    b.axhline(0,color='0.3',lw=0.8,ls=':')
    b.set(xlabel=r'$\phi/\pi$',
          ylabel=r'$10^3 \Delta \theta$')
    b.legend(loc='best',frameon=False)
    b.text(-0.16,1.02,'(b)',transform=b.transAxes,fontsize=14)
    fig.subplots_adjust(left=0.105,right=0.98,bottom=0.21,top=0.91,wspace=0.13)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png','.pdf'):
        fig.savefig(str(args.out)+ext,dpi=args.dpi,bbox_inches='tight')
    plt.close(fig)
    print('Saved',args.out+'.png', 'and',args.out+'.pdf')
    print('Harmonics:',harmonics)

if __name__=='__main__':
    main()
