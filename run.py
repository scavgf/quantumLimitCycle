#!/usr/bin/env python3
"""Build manuscript figures from shared, cached Julia data (Python 3.11+)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parent
DATA_FILES = ['approach.csv', 'cycle.csv', 'husimi_q.csv', 'photon_pop.csv',
              'spectrum.csv', 'weights.csv', 'fs_selected.csv', 'fidelity.csv',
              'metric_time.csv', 'bures_absolute.csv', 'bures_repaired_mask.csv',
              'metadata.txt', 'theta_comparison.csv', 'harmonic_amplitudes.csv',
              'summary.txt', 'comparison_parameters.txt', 'positivity_report.txt',
              'reference_state_real.csv', 'reference_state_imag.csv', 'reference_harmonics.csv',
              'bures_min_eigenvalue.csv', 'bures_trace_error.csv', 'bures_hermiticity_error.csv']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(paths, args):
    return {'inputs': {str(p.resolve()): digest(p) for p in paths}, 'arguments': args}


def current(stamp, sig, outputs):
    try:
        prior = json.loads(stamp.read_text())
        return prior['signature'] == sig and prior['outputs'] == {
            str(p.resolve()): digest(p) for p in outputs}
    except (OSError, ValueError, KeyError):
        return False


def checkpoint(stamp, sig, outputs):
    value = {'signature': sig, 'outputs': {str(p.resolve()): digest(p) for p in outputs}}
    tmp = stamp.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(stamp)


def execute(command, dry):
    import shlex
    print('RUN ' + shlex.join(map(str, command)), flush=True)
    if not dry:
        env = dict(os.environ, MPLBACKEND='Agg')
        subprocess.run(list(map(str, command)), cwd=ROOT, check=True, env=env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'config.toml')
    p.add_argument('--work', type=Path, default=ROOT/'results')
    p.add_argument('--figures', nargs='+', type=int, choices=range(1, 7), default=list(range(1, 7)))
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--data-only', action='store_true')
    modes.add_argument('--plot-only', action='store_true', help='Require complete data matching current config/code')
    p.add_argument('--force-plots', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--julia', default='julia', help='Julia executable name or path')
    args = p.parse_args()
    config_path = args.config.resolve()
    config = tomllib.loads(config_path.read_text())
    work = args.work.resolve(); data = work/'data'; figures = work/'figures'
    stamps = work/'build'
    if not args.dry_run:
        for path in (data, figures, stamps):
            path.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT/'src').glob('*.jl')) + [ROOT/'Project.toml', config_path]
    sig = signature(sources, ['data-v1'])
    files = [data/name for name in DATA_FILES]
    data_ok = current(stamps/'data.json', sig, files)
    if not data_ok:
        if args.plot_only:
            p.error('Data are missing, changed, or stale for this config/code. Run without --plot-only first.')
        if not args.dry_run and shutil.which(args.julia) is None:
            p.error('Julia was not found. Install Julia 1.10+ or pass --julia /path/to/julia.')
        execute([args.julia, '--project='+str(ROOT), ROOT/'src/calculate.jl', config_path, work], args.dry_run)
        if not args.dry_run:
            checkpoint(stamps/'data.json', sig, files)
    else:
        print('REUSE all numerical data (inputs and output hashes match)', flush=True)
    if args.data_only:
        return
    omega = str(config['model']['omega0']); bt = str(config['observations']['bures_time'])
    epsilon = str(config['model']['epsilon'])
    hs = list(map(str, config['observations']['phase_harmonics']))
    commands = {
        1: ['--classical', data, '--geometry', data, '--time', bt, '--omega', omega, '--epsilon', epsilon, '--output', figures/'Fig1'],
        2: [data, '--omega', omega, '--times', ','.join(map(str, config['observations']['times'])), '--out', figures/'Fig2'],
        3: [data, '--fitmax', '5', '--out', figures/'Fig3'],
        4: [data, '--out', figures/'Fig4', '--harmonics', *hs],
        5: [data, '--out', figures/'Fig5', '--Delta', omega, '--epsilon', epsilon],
        6: [data, '--out', figures/'Fig6'],
    }
    needed = {1:['approach.csv','cycle.csv','husimi_q.csv','fs_selected.csv','bures_absolute.csv','bures_repaired_mask.csv','metadata.txt'],
              2:['weights.csv','fidelity.csv','metric_time.csv','fs_selected.csv','spectrum.csv'],
              3:['spectrum.csv'], 4:['theta_comparison.csv'], 5:['husimi_q.csv'],
              6:['positivity_report.txt','reference_harmonics.csv','spectrum.csv','metric_time.csv']}
    for number in sorted(set(args.figures)):
        script = ROOT/'plots'/f'fig{number}.py'
        command = [sys.executable, script, *commands[number]]
        outputs = [figures/f'Fig{number}.{ext}' for ext in ('pdf','png')]
        if number == 2:
            outputs.append(figures/'Fig2_geometry_frequency.csv')
        if number == 5:
            outputs.extend([figures/'Fig5_p_theta.csv', figures/'Fig5_phase_comparison.csv'])
        if number == 6:
            outputs.extend([figures/'Fig6_harmonics.csv', figures/'Fig6_time.csv'])
        stamp = stamps/f'fig{number}.json'
        paths = [script, ROOT/'run.py', ROOT/'requirements.txt'] + [data/f for f in needed[number]]
        if number == 1:
            paths.append(ROOT/'plots'/'fig5.py')
        if args.dry_run and not all(path.is_file() for path in paths):
            execute(command, True)
            continue
        plot_sig = signature(paths, list(map(str, command)))
        if not args.force_plots and current(stamp, plot_sig, outputs):
            print(f'REUSE Fig{number}', flush=True)
            continue
        execute(command, args.dry_run)
        if not args.dry_run:
            checkpoint(stamp, plot_sig, outputs)


if __name__ == '__main__':
    main()
