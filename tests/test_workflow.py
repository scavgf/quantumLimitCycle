"""Cache and plotting checks using synthetic inputs, not scientific validation."""
import importlib.util
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('workflow', ROOT/'run.py')
workflow = importlib.util.module_from_spec(spec); spec.loader.exec_module(workflow)


def table(path, header, values):
    np.savetxt(path, values, delimiter=',', header=header, comments='')


def fixture(data):
    times = np.array([0., 1., 4., 5.]); ell = np.arange(-5, 6)
    z = -.003*ell**2 - .8j*ell
    weights = np.exp(-ell[:, None]**2/8 + 2*z.real[:, None]*times)
    weights /= weights.sum(axis=0)
    metric = ((ell[:,None]**2)*weights).sum(axis=0)
    table(data/'spectrum.csv','l,Re_z,Im_z,gamma',np.c_[ell,z.real,z.imag,-z.real])
    table(data/'weights.csv','l,t_0,t_1,t_4,t_5',np.c_[ell,weights])
    table(data/'fs_selected.csv','t,g_FS',np.c_[times,metric])
    t = np.linspace(0,20,201); wt=np.exp(-ell[:,None]**2/8+2*z.real[:,None]*t); wt/=wt.sum(axis=0)
    table(data/'metric_time.csv','t,g_FS',np.c_[t,(ell[:,None]**2*wt).sum(axis=0)])
    tau = 1.
    gaussian = np.sqrt(weights[:,0])
    projected_seed = gaussian * np.exp(-tau*z)
    table(data/'reference_harmonics.csv',
          'l,gaussian_weight,log_gaussian_envelope,raw_c_real,raw_c_imag,gaussian_c_real,gaussian_c_imag',
          np.c_[ell, weights[:,0], -ell**2/8, projected_seed.real, projected_seed.imag,
                gaussian, np.zeros(len(ell))])
    (data/'positivity_report.txt').write_text(
        'certified_times_origin=Gaussian start\n'
        f'gaussian_start_projection_time={tau}\n'
        f'projected_variance_at_match_time={metric[0]}\n'
        f'selected_gaussian_variance_at_match_time={metric[0]}\n'
        'weight_total_variation_at_match_time=0.0\n')
    d=np.linspace(-1,1,101)
    fidelity=abs(np.exp(1j*np.pi*d[:,None]*ell)@weights)**2
    table(data/'fidelity.csv','delta_theta_over_pi,t_0,t_1,t_4,t_5',np.c_[d,fidelity])
    th=np.linspace(0,2,48,endpoint=False)
    table(data/'bures_absolute.csv','theta_over_pi,t_4',np.c_[th,np.ones(48)*metric[2]/2])
    table(data/'bures_repaired_mask.csv','theta_over_pi,t_4',np.c_[th,np.zeros(48)])
    a=np.linspace(0,2*np.pi,121); x=5*np.cos(a); y=3*np.sin(a)
    np.savetxt(data/'cycle.csv',np.c_[a,x,y,np.ones(121)/(8*np.pi)],delimiter=',')
    np.savetxt(data/'approach.csv',np.c_[a,x,y,np.zeros(121)],delimiter=',')
    xx,pp=np.meshgrid(np.linspace(-7,7,41),np.linspace(-7,7,41))
    q=np.exp(-((xx/5)**2+(pp/3)**2-1)**2*12)
    np.savetxt(data/'husimi_q.csv',np.c_[xx.ravel(),pp.ravel(),q.ravel()],delimiter=',')
    table(data/'photon_pop.csv','n,p',np.c_[np.arange(48),np.ones(48)/48])
    phase=-a; phi=np.unwrap(np.arctan2(y,x))
    columns=[a,phi,phase,-.8*a,x,y]; names=['t','phi_unwrapped','theta_classical','theta_uniform_time','x','p']
    for l in (1,3,5):
        error=.01*np.sin(2*a)/l
        columns.extend([phase+error,error,np.ones(121),np.ones(121)])
        names.extend([f'theta_quantum_l{l}',f'phase_error_l{l}',f'amplitude_l{l}',f'amplitude_ratio_l{l}'])
    table(data/'theta_comparison.csv',','.join(names),np.array(columns).T)
    (data/'metadata.txt').write_text('Delta=1.0\nepsilon=0.6\nomega_cycle=0.8\nD_fit=0.006\n')
    for name in workflow.DATA_FILES:
        if not (data/name).exists():
            (data/name).write_text('synthetic fixture\n')


class WorkflowTests(unittest.TestCase):
    def test_cache_invalidates_inputs_outputs_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; out=root/'out'; stamp=root/'stamp'
            source.write_text('a');out.write_text('b')
            sig=workflow.signature([source],['run'])
            workflow.checkpoint(stamp,sig,[out])
            self.assertTrue(workflow.current(stamp,sig,[out]))
            source.write_text('c')
            self.assertFalse(workflow.current(stamp,workflow.signature([source],['run']),[out]))
            out.write_text('changed')
            self.assertFalse(workflow.current(stamp,sig,[out]))
            out.unlink();self.assertFalse(workflow.current(stamp,sig,[out]))

    def test_all_plots_and_second_run_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);data=work/'data';data.mkdir();(work/'build').mkdir()
            fixture(data)
            config=ROOT/'config_smoke.toml'
            sources=sorted((ROOT/'src').glob('*.jl'))+[ROOT/'Project.toml',config]
            workflow.checkpoint(work/'build/data.json',workflow.signature(sources,['data-v1']),[data/n for n in workflow.DATA_FILES])
            command=[sys.executable,str(ROOT/'run.py'),'--config',str(config),'--work',str(work),'--plot-only']
            first=subprocess.run(command,capture_output=True,text=True,env=dict(os.environ,MPLBACKEND='Agg'))
            self.assertEqual(first.returncode,0,first.stdout+'\n'+first.stderr)
            self.assertEqual(len(list((work/'figures').glob('*.pdf'))),6)
            for name in ('Fig6_harmonics.csv', 'Fig6_time.csv'):
                self.assertTrue((work/'figures'/name).is_file())
            second=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(second.returncode,0,second.stderr)
            for n in range(1,7):self.assertIn(f'REUSE Fig{n}',second.stdout)
            self.assertNotIn('RUN ',second.stdout)
            # The new derived metric CSV must also participate in the build cache.
            csv = work/'figures/Fig2_geometry_frequency.csv'
            self.assertTrue(csv.is_file())
            csv.unlink()
            repaired=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(repaired.returncode,0,repaired.stderr)
            self.assertTrue(csv.is_file())
            self.assertEqual(repaired.stdout.count('RUN '),1)
            for n in (1,3,4,5,6): self.assertIn(f'REUSE Fig{n}',repaired.stdout)
            # Keep an optional contact sheet outside the deliverable for visual QA.
            if os.environ.get('LC_QA_DIR'):
                import shutil
                out=Path(os.environ['LC_QA_DIR']);out.mkdir(parents=True,exist_ok=True)
                for p in (work/'figures').glob('*.png'):shutil.copy2(p,out/p.name)

if __name__=='__main__':unittest.main()
