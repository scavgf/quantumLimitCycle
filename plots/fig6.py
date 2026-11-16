#!/usr/bin/env python3
"""Compare a shifted-time Gaussian fit with its evolved Liouvillian projection.

Usage: python3 plots/fig6.py results/data
Optional: --basis results/cache/basis_<hash>.jls for exact Hilbert–Schmidt errors.
The basis must be from the same run and trusted. No eigensolver is called.
"""
import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Julia reads the trusted cache using its native serialization format. It
# returns the Gram matrix of the normalized modes in the exported phase gauge.
JULIA_GRAM = """
using Serialization, DelimitedFiles, LinearAlgebra
basis = open(deserialize, ARGS[1])
folder, output = ARGS[2:3]
harmonics = readdlm(joinpath(folder, "reference_harmonics.csv"), ',', Float64; skipstart=1)
spectrum = readdlm(joinpath(folder, "spectrum.csv"), ',', Float64; skipstart=1)
order = sortperm(spectrum[:, 1])
spectrum = spectrum[order, :]
basis.ell == spectrum[:, 1] || error("Basis harmonics do not match spectrum.csv")
isapprox(basis.Z, complex.(spectrum[:, 2], spectrum[:, 3]); rtol=1e-10, atol=1e-12) ||
    error("Basis eigenvalues do not match spectrum.csv")
sizes = [norm(col) for col in eachcol(basis.R)]
R = basis.R ./ reshape(sizes, 1, :)
harmonics = harmonics[sortperm(harmonics[:, 1]), :]
c = complex.(harmonics[:, 6], harmonics[:, 7])
reference = complex.(readdlm(joinpath(folder, "reference_state_real.csv"), ',', Float64),
                    readdlm(joinpath(folder, "reference_state_imag.csv"), ',', Float64))
isapprox(R*c, vec(reference); rtol=1e-8, atol=1e-11) ||
    error("Cached right modes do not reconstruct the exported Gaussian reference")
G = R' * R
writedlm(output, hcat(real.(G), imag.(G)), ',')
"""


def read_table(path):
    table = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    table = np.atleast_1d(table)
    if table.size == 0 or any(not np.all(np.isfinite(table[k])) for k in table.dtype.names):
        raise ValueError(f"Missing or nonfinite values in {path}")
    return table


def evolved_weights(initial, decay, time):
    log_initial = np.full(initial.shape, -np.inf)
    np.log(initial, out=log_initial, where=initial > 0)
    logs = log_initial[None, :] + 2 * time[:, None] * decay[None, :]
    w = np.exp(logs - logs.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


def variance(weights, harmonics):
    mean = weights @ harmonics
    return np.sum(weights * (harmonics - mean[:, None])**2, axis=1)


def hs_norm(coefficients, gram):
    squared = np.einsum("ti,ij,tj->t", coefficients.conj(), gram, coefficients).real
    return np.sqrt(np.maximum(squared, 0.0))


def main():
    plt.rcParams['font.family']='Times New Roman'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Shifted-time run's results/data directory")
    parser.add_argument("--out", type=Path, default=Path("Fig6"),
                        help="Output stem for PDF, PNG and two CSV files")
    parser.add_argument("--basis", type=Path, help="Trusted basis_*.jls from this run")
    parser.add_argument("--julia", default="julia", help="Julia executable used to read --basis")
    parser.add_argument("--label", default="", help="Optional figure subtitle")
    args = parser.parse_args()
    folder = args.data.resolve()
    report = dict(line.split("=", 1) for line in
                  (folder / "positivity_report.txt").read_text().splitlines()
                  if "=" in line)
    if report.get("certified_times_origin") != "Gaussian start":
        parser.error("Use the data directory from a shifted-time Gaussian run.")
    tau = float(report["gaussian_start_projection_time"])
    reference = read_table(folder / "reference_harmonics.csv")
    spectrum = read_table(folder / "spectrum.csv")
    reference.sort(order="l")
    spectrum.sort(order="l")
    ell = reference["l"]
    if len(np.unique(ell)) != len(ell) or not np.array_equal(ell, spectrum["l"]):
        parser.error("The reference and spectrum harmonic labels must match uniquely.")
    z = spectrum["Re_z"] + 1j * spectrum["Im_z"]
    c_raw = reference["raw_c_real"] + 1j * reference["raw_c_imag"]
    c_g = reference["gaussian_c_real"] + 1j * reference["gaussian_c_imag"]
    p_initial = reference["gaussian_weight"]

    metric = read_table(folder / "metric_time.csv")
    metric.sort(order="t")
    s = metric["t"]
    if s[0] != 0 or np.any(np.diff(s) <= 0):
        parser.error("metric_time.csv must contain unique times starting at zero.")
    q = evolved_weights(np.abs(c_raw)**2, z.real, tau + s)
    p = evolved_weights(p_initial, z.real, s)
    fs_proj, fs_g = variance(q, ell), variance(p, ell)
    tv = 0.5 * np.abs(p - q).sum(axis=1)
    if not np.allclose(fs_g, metric["g_FS"], rtol=1e-7, atol=1e-12):
        parser.error("The Gaussian weights do not reproduce metric_time.csv.")
    expected = [float(report[key]) for key in (
        "projected_variance_at_match_time", "selected_gaussian_variance_at_match_time",
        "weight_total_variation_at_match_time")]
    if not np.allclose([fs_proj[0], fs_g[0], tv[0]], expected, rtol=1e-7, atol=1e-12):
        parser.error("The initial comparison does not reproduce positivity_report.txt.")
    fs_error = np.abs(fs_g - fs_proj)
    fs_rel = np.divide(fs_error, fs_proj, out=np.full_like(fs_proj, np.nan),
                       where=fs_proj > 0)

    output = args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    series = [s, tau+s, fs_proj, fs_g, fs_error, fs_rel, tv]
    headers = ["s", "projection_time", "fs_projected", "fs_gaussian",
               "fs_absolute_error", "fs_relative_error", "total_variation"]
    if args.basis:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "gram.csv"
            subprocess.run([args.julia, "--startup-file=no", "-e", JULIA_GRAM,
                            str(args.basis.resolve()), str(folder), str(path)], check=True)
            blocks = np.loadtxt(path, delimiter=",", ndmin=2)
        gram = blocks[:, :len(ell)] + 1j * blocks[:, len(ell):]
        projected = c_raw[None, :] * np.exp((tau+s[:, None]) * z)
        gaussian = c_g[None, :] * np.exp(s[:, None] * z)
        hs_error = hs_norm(gaussian - projected, gram)
        hs_relative = hs_error / hs_norm(projected, gram)
        series += [hs_error, hs_relative]
        headers += ["hs_absolute_error", "hs_relative_error"]
        print(f"At s=0: HS error = {hs_error[0]:.8g}, relative HS error = {hs_relative[0]:.6%}")

    np.savetxt(f"{output}_time.csv", np.column_stack(series), delimiter=",",
               header=",".join(headers), comments="")
    np.savetxt(f"{output}_harmonics.csv", np.column_stack([ell, q[0], p[0], p[0]-q[0]]),
               delimiter=",", header="l,q_projected_at_tau,p_gaussian_at_zero,p_minus_q",
               comments="")
    print(f"At s=0: TV = {tv[0]:.6%}, FS absolute error = {fs_error[0]:.8g}")
    print(f"Over the saved range: max TV = {tv.max():.6%}, max FS absolute error = {fs_error.max():.8g}")

    blue, orange = "#285880", "#bf5b32"
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 18,
                         'xtick.labelsize':18,
                         'ytick.labelsize':18,
                         "font.family": 'Times New Roman',
                          'mathtext.fontset': 'stix',
                         "axes.spines.top": True,
                         "axes.spines.right": True, "savefig.facecolor": "white"})
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.55), sharex=True, layout="constrained")
    ax = axs[0]
    ax.plot(ell, q[0], "o-", ms=3.5, color=blue, label=r"$q_l(\tau)$")
    ax.plot(ell, p[0], "s--", ms=3.5, color=orange, label="$p_l(0)$")
    ax.set(xlabel=r"$l$", ylabel="Normalized weight", title="(a) Harmonic weights")
    ax.legend(frameon=False, fontsize=18)
    ax = axs[1]
    ax.axhline(0, color="#6d7882", lw=0.75)
    ax.bar(ell, 10**3*(p[0]-q[0]), width=0.75, color=np.where(p[0]>=q[0], orange, blue))
    ax.set(xlabel=r"$l$", ylabel=r"$10^3\delta p_l$",
           title=f"(b) Weight residuals  |  TV = {tv[0]:.3%}")
    for ax in axs:
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.savefig(f"{output}.pdf")
    fig.savefig(f"{output}.png", dpi=230)
    plt.close(fig)


if __name__ == "__main__":
    main()
