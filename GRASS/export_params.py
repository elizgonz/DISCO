"""
export_disco_params.py
Export DISCO Fe5250 model parameters to HDF5 using DISCO's LineProfileGenerator.
"""
import sys
import os
import numpy as np
import h5py
from scipy.stats import skewnorm as scipy_skewnorm
from scipy.optimize import brentq

# Path setup
DISCO_ROOT = "/storage/home/efg5335/work/sw/DISCO"
sys.path.insert(0, os.path.join(DISCO_ROOT, "line_creation"))

from make_lines import LineProfileGenerator, exponential

LINE       = "Fe5250"
MU_MIN     = 0.10
MU_MAX     = 1.00
N_MU_DENSE = 512

print(f"Loading LineProfileGenerator for {LINE}...")
gen = LineProfileGenerator(LINE)

mu_dense  = np.linspace(MU_MIN, MU_MAX, N_MU_DENSE, dtype=np.float32)
deg_dense = np.degrees(np.arccos(mu_dense))

components = {
    "GT":  (gen.mean_profile_gt,  gen.eigenprofiles_gt,  gen.coef_interp_gt),
    "OGR": (gen.mean_profile_ogr, gen.eigenprofiles_ogr, gen.coef_interp_ogr),
    "IgL": (gen.mean_profile_igl, gen.eigenprofiles_igl, gen.coef_interp_igl),
}

print("Evaluating PCA spline interpolators...")
pca_coeff_grids = {}
for cname, (_, _, coef_interp) in components.items():
    grid = np.stack([[interp(d) for d in deg_dense] for interp in coef_interp])  # (n_pca, N_MU_DENSE)
    pca_coeff_grids[cname] = grid.astype(np.float32)

n_wave = len(gen.wl)
n_pca  = gen.eigenprofiles_gt.shape[0]

def skewnorm_skewness(alpha):
    delta = alpha / np.sqrt(1.0 + alpha**2)
    mu_z  = delta * np.sqrt(2.0 / np.pi)
    return (4.0 - np.pi) / 2.0 * mu_z**3 / (1.0 - mu_z**2)**1.5

def get_skewnorm_params(med_coeffs, std_coeffs, skew_coeffs, deg_arr, is_gt=True):
    n = len(deg_arr)
    alphas = np.zeros(n, dtype=np.float32)
    locs   = np.zeros(n, dtype=np.float32)
    scales = np.zeros(n, dtype=np.float32)

    poly_med  = np.poly1d(med_coeffs)
    poly_skew = np.poly1d(skew_coeffs)

    for i, deg in enumerate(deg_arr):
        med = poly_med(deg)
        if is_gt:
            std = exponential(deg, *std_coeffs) * med
        else:
            std = exponential(deg, *std_coeffs)

        skew_val = float(poly_skew(deg))
        skew_val = np.clip(skew_val, -0.98, 0.98)

        alpha = brentq(lambda a: skewnorm_skewness(a) - skew_val, -50.0, 50.0)
        delta = alpha / np.sqrt(1.0 + alpha**2)
        mu_z  = delta * np.sqrt(2.0 / np.pi)
        scale = std / np.sqrt(1.0 - mu_z**2)
        loc   = med - scale * float(scipy_skewnorm(alpha).median())

        alphas[i], locs[i], scales[i] = alpha, loc, scale

    return alphas, locs, scales

print("Evaluating filling factor parameters...")
e1_alpha, e1_loc, e1_scale = get_skewnorm_params(
    gen.gt_med_coeffs, gen.gt_std_coeffs, gen.gt_skew_coeffs, deg_dense, is_gt=True)

e2_alpha, e2_loc, e2_scale = get_skewnorm_params(
    gen.ratio_med_coeffs, gen.ratio_std_coeffs, gen.ratio_skew_coeffs, deg_dense, is_gt=False)

out_path = "/storage/home/efg5335/work/sw/DISCO/GRASS/disco_fe5250_params.h5"
print(f"Writing {out_path}...")

with h5py.File(out_path, "w") as f:
    f.attrs["line"]   = LINE
    f.attrs["mu_min"] = float(MU_MIN)
    f.attrs["mu_max"] = float(MU_MAX)
    f.attrs["n_mu"]   = N_MU_DENSE
    f.attrs["n_wave"] = n_wave
    f.attrs["n_pca"]  = n_pca

    f.create_dataset("wavelength", data=gen.wl.astype(np.float32))
    f.create_dataset("mu_grid",    data=mu_dense)

    for cname, (mean, phis, _) in components.items():
        grp = f.create_group(f"pca/{cname}")
        grp.create_dataset("mean_profile",   data=mean.astype(np.float32))
        grp.create_dataset("eigenprofiles",  data=phis.astype(np.float32))
        grp.create_dataset("pca_coeff_grid", data=pca_coeff_grids[cname])

    d = f.create_group("distributions")
    for name, alpha, loc, scale in [
        ("epsilon1", e1_alpha, e1_loc, e1_scale),
        ("epsilon2", e2_alpha, e2_loc, e2_scale),
    ]:
        g = d.create_group(name)
        g.create_dataset("alpha", data=alpha)
        g.create_dataset("zeta",  data=loc)
        g.create_dataset("omega", data=scale)

print("Export completed successfully.")