"""
export_disco_params.py
──────────────────────────────────────────────────────────────────────────────
Exports DISCO Fe I 5250 parameters to HDF5 for the GRASS-E GPU pipeline.

Run from anywhere — DISCO_ROOT is resolved relative to this file's location
(assumed to live in DISCO/GRASS/):

    python export_disco_params.py
    python export_disco_params.py --output my_params.h5

What this script does
─────────────────────
1. Instantiates LineProfileGenerator("Fe5250"), which loads all pre-trained
   .npy files (mean profiles, eigenprofiles, spline interpolators,
   filling-factor polynomial coefficients) from DISCO/line_creation/data/.

2. Evaluates the pre-fitted PCA spline interpolators (coef_interp_*) on a
   dense N_MU_DENSE uniform mu grid to get pca_coeff_grid per component.

3. Converts DISCO's filling-factor polynomial coefficients (median, std, skew
   as functions of viewing angle) into skewnorm(alpha, loc, scale) parameters
   on the same dense mu grid.

4. Writes everything to a compact HDF5 file (~few MB). The Julia GPU kernel
   only needs a single linear interpolation at runtime — no spline/polynomial
   evaluation on device.
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import numpy as np
import h5py
from scipy.stats import skewnorm as scipy_skewnorm
from scipy.optimize import brentq

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

LINE       = "Fe5250"
MU_MIN     = 0.10      # avoid exact limb singularity
MU_MAX     = 1.00
N_MU_DENSE = 512       # dense runtime grid; 512 → <0.1% linear-interp error

# ─────────────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────────────

DISCO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(DISCO_ROOT, "line_creation"))

from make_lines import LineProfileGenerator

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Export DISCO Fe5250 params to HDF5")
parser.add_argument("--output", default="disco_fe5250_params.h5")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Instantiate — loads all pre-trained .npy files from DISCO/line_creation/data/
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading LineProfileGenerator for {LINE} ...")
gen = LineProfileGenerator(LINE)
print(f"  wavelength points : {len(gen.wl)}")
print(f"  n_pca (GT)        : {gen.eigenprofiles_gt.shape[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# Dense mu grid — DISCO's interpolators expect viewing angle in degrees
# ─────────────────────────────────────────────────────────────────────────────

mu_dense  = np.linspace(MU_MIN, MU_MAX, N_MU_DENSE, dtype=np.float32)
deg_dense = np.degrees(np.arccos(mu_dense))   # DISCO convention: 0=disk center, 90=limb

# ─────────────────────────────────────────────────────────────────────────────
# PCA coefficient grids
# Each coef_interp_* is a pre-fitted scipy interpolator: f(deg) -> (n_pca,)
# ─────────────────────────────────────────────────────────────────────────────

print("Evaluating PCA spline interpolators on dense mu grid ...")

components = {
    "GT":  (gen.mean_profile_gt,  gen.eigenprofiles_gt,  gen.coef_interp_gt),
    "OGR": (gen.mean_profile_ogr, gen.eigenprofiles_ogr, gen.coef_interp_ogr),
    "IgL": (gen.mean_profile_igl, gen.eigenprofiles_igl, gen.coef_interp_igl),
}

pca_coeff_grids = {}
for cname, (_, _, coef_interp) in components.items():
    # coef_interp(deg) returns shape (n_pca,) — stack over dense grid
    grid = np.stack([[interp(d) for d in deg_dense] for interp in coef_interp])
    pca_coeff_grids[cname] = grid.astype(np.float32)
    print(f"  {cname}: pca_coeff_grid shape = {grid.shape}")

n_wave = len(gen.wl)
n_pca  = gen.eigenprofiles_gt.shape[0]

# ─────────────────────────────────────────────────────────────────────────────
# Filling-factor skewnorm parameters
#
# DISCO stores polynomial coefficients (np.polyval convention) for the
# median, normalised std, and skewness of each filling factor as a function
# of viewing angle in degrees.  We convert these to skewnorm(alpha, loc, scale)
# on the dense mu grid so the GPU kernel only needs to linearly interpolate
# three scalars per filling factor.
# ─────────────────────────────────────────────────────────────────────────────

def _skewnorm_skewness(alpha):
    """Analytical skewness of a skewnorm(alpha, 0, 1) distribution."""
    delta = alpha / np.sqrt(1.0 + alpha**2)
    mu_z  = delta * np.sqrt(2.0 / np.pi)
    return (4.0 - np.pi) / 2.0 * mu_z**3 / (1.0 - mu_z**2)**1.5


def poly_to_skewnorm(med_coeffs, std_coeffs, skew_coeffs, deg_arr):
    """
    Convert DISCO polynomial coefficients to skewnorm(alpha, loc, scale)
    evaluated at each angle in deg_arr.

    Returns three float32 arrays of length len(deg_arr).
    """
    n      = len(deg_arr)
    alphas = np.zeros(n, dtype=np.float32)
    locs   = np.zeros(n, dtype=np.float32)
    scales = np.zeros(n, dtype=np.float32)

    for i, deg in enumerate(deg_arr):
        med      = np.polyval(med_coeffs,  deg)
        std      = np.polyval(std_coeffs,  deg)
        skew_val = float(np.polyval(skew_coeffs, deg))

        # Skewnorm skewness is bounded in (-1, 1) — clip conservatively
        skew_val = np.clip(skew_val, -0.98, 0.98)

        # Solve for alpha numerically
        alpha = brentq(lambda a: _skewnorm_skewness(a) - skew_val, -50.0, 50.0)

        # Derive scale from std: std(skewnorm) = scale * sqrt(1 - (delta*sqrt(2/pi))^2)
        delta = alpha / np.sqrt(1.0 + alpha**2)
        mu_z  = delta * np.sqrt(2.0 / np.pi)
        scale = std / np.sqrt(1.0 - mu_z**2)

        # Derive loc from median: median = loc + scale * skewnorm.median()
        loc = med - scale * float(scipy_skewnorm(alpha).median())

        alphas[i], locs[i], scales[i] = alpha, loc, scale

    return alphas, locs, scales


print("Converting filling-factor polynomials to skewnorm parameters ...")

e1_alpha, e1_loc, e1_scale = poly_to_skewnorm(
    gen.gt_med_coeffs, gen.gt_std_coeffs, gen.gt_skew_coeffs, deg_dense)

e2_alpha, e2_loc, e2_scale = poly_to_skewnorm(
    gen.ratio_med_coeffs, gen.ratio_std_coeffs, gen.ratio_skew_coeffs, deg_dense)

print("  epsilon1 (GT)    alpha range:", e1_alpha.min(), "→", e1_alpha.max())
print("  epsilon2 (ratio) alpha range:", e2_alpha.min(), "→", e2_alpha.max())

# ─────────────────────────────────────────────────────────────────────────────
# Write HDF5
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nWriting {args.output} ...")

with h5py.File(args.output, "w") as f:
    # Top-level metadata
    f.attrs["line"]   = LINE
    f.attrs["mu_min"] = float(MU_MIN)
    f.attrs["mu_max"] = float(MU_MAX)
    f.attrs["n_mu"]   = N_MU_DENSE
    f.attrs["n_wave"] = n_wave
    f.attrs["n_pca"]  = n_pca

    # Axes
    f.create_dataset("wavelength", data=gen.wl.astype(np.float32))
    f.create_dataset("mu_grid",    data=mu_dense)

    # PCA — one group per granulation component
    for cname, (mean, phis, _) in components.items():
        grp = f.create_group(f"pca/{cname}")
        grp.create_dataset("mean_profile",   data=mean.astype(np.float32))
        grp.create_dataset("eigenprofiles",  data=phis.astype(np.float32))
        grp.create_dataset("pca_coeff_grid", data=pca_coeff_grids[cname])

    # Filling-factor distribution parameters on the dense mu grid
    for name, alpha, loc, scale in [
        ("epsilon1", e1_alpha, e1_loc, e1_scale),
        ("epsilon2", e2_alpha, e2_loc, e2_scale),
    ]:
        g = f.create_group(f"distributions/{name}")
        g.create_dataset("alpha", data=alpha)
        g.create_dataset("zeta",  data=loc)
        g.create_dataset("omega", data=scale)

print(f"Done. Output: {args.output}")
print(f"  wavelength : {n_wave} points")
print(f"  mu grid    : {N_MU_DENSE} points  [{MU_MIN}, {MU_MAX}]")
print(f"  n_pca      : {n_pca} per component")
