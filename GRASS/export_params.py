"""
export_disco_params.py
──────────────────────────────────────────────────────────────────────────────
Exports DISCO parameters to HDF5 for the GRASS-E GPU pipeline.

What this script does
─────────────────────
1. Instantiates LineProfileGenerator(line), which loads pre-trained .npy
   files from DISCO/line_creation/data/line/.

2. Evaluates the PCA spline interpolators (coef_interp_*) on a dense 4096-point
   uniform mu grid covering the model's valid angular domain (0–83 degrees).

3. Extracts DISCO's filling-factor distribution parameters (alpha, zeta, omega)
   and evaluates the exact median (0.5 quantile / 50th percentile) filling factor
   on the dense grid.

4. Writes everything to HDF5 so the GPU kernel can perform fast linear
   interpolation on device without extrapolating beyond the fitted model.
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import numpy as np
import h5py
from scipy.stats import skewnorm as scipy_skewnorm

# Path setup — resolves relative to DISCO root
DISCO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(DISCO_ROOT, "line_creation"))

from make_lines import LineProfileGenerator, exponential

LINE = "Fe5250"
MAX_DEG = 83.0
MU_MIN = float(np.nextafter(
    np.float32(np.cos(np.deg2rad(MAX_DEG))), np.float32(1.0)))
MU_MAX = 1.00
N_MU_DENSE = 4096

def get_distribution_params(med_coeffs, std_coeffs, skew_coeffs, deg_arr, is_gt=True):
    """
    Extract exact skewnorm parameters (alpha, zeta, omega) and compute the 50th percentile 
    (0.5 quantile) filling factor across deg_arr.
    """
    poly_med  = np.poly1d(med_coeffs)
    poly_skew = np.poly1d(skew_coeffs)

    alphas = poly_skew(deg_arr).astype(np.float32)
    zetas  = poly_med(deg_arr).astype(np.float32)

    if is_gt:
        omegas = (exponential(deg_arr, *std_coeffs) * zetas).astype(np.float32)
    else:
        omegas = exponential(deg_arr, *std_coeffs).astype(np.float32)

    # Compute median (0.5 quantile / 50th percentile) filling factor at each angle
    medians = np.zeros(len(deg_arr), dtype=np.float32)
    for i, (a, z, w) in enumerate(zip(alphas, zetas, omegas)):
        medians[i] = float(scipy_skewnorm(a, loc=z, scale=w).ppf(0.5))

    return alphas, zetas, omegas, medians

def export_params(output_path=None, generator=None, line=LINE):
    """Export DISCO's model in the format consumed by GRASS."""
    if generator is not None and generator.line != line:
        raise ValueError(
            f"Generator line {generator.line!r} does not match requested line {line!r}")
    print(f"Loading LineProfileGenerator for {line}...")
    gen = generator if generator is not None else LineProfileGenerator(line)
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "data", f"disco_{line}_params.h5")

    mu_dense = np.linspace(MU_MIN, MU_MAX, N_MU_DENSE, dtype=np.float32)
    deg_dense = np.minimum(
        np.degrees(np.arccos(mu_dense.astype(np.float64))), MAX_DEG)

    components = {
        "GT": (gen.mean_profile_gt, gen.eigenprofiles_gt, gen.coef_interp_gt),
        "OGR": (gen.mean_profile_ogr, gen.eigenprofiles_ogr, gen.coef_interp_ogr),
        "IgL": (gen.mean_profile_igl, gen.eigenprofiles_igl, gen.coef_interp_igl),
    }

    print("Evaluating PCA spline interpolators on dense mu grid...")
    pca_coeff_grids = {
        cname: np.stack([[interp(d) for d in deg_dense]
                         for interp in coef_interp]).astype(np.float32)
        for cname, (_, _, coef_interp) in components.items()
    }

    n_wave = len(gen.wl)
    n_pca = gen.eigenprofiles_gt.shape[0]

    print("Evaluating filling factor distribution parameters...")
    e1_alpha, e1_zeta, e1_omega, e1_med = get_distribution_params(
        gen.gt_med_coeffs, gen.gt_std_coeffs, gen.gt_skew_coeffs,
        deg_dense, is_gt=True)
    e2_alpha, e2_zeta, e2_omega, e2_med = get_distribution_params(
        gen.ratio_med_coeffs, gen.ratio_std_coeffs, gen.ratio_skew_coeffs,
        deg_dense, is_gt=False)

    print(f"Writing {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.attrs["line"] = line
        f.attrs["mu_min"] = float(mu_dense[0])
        f.attrs["mu_max"] = float(mu_dense[-1])
        f.attrs["n_mu"] = N_MU_DENSE
        f.attrs["n_wave"] = n_wave
        f.attrs["n_pca"] = n_pca

        f.create_dataset("wavelength", data=gen.wl.astype(np.float32))
        f.create_dataset("mu_grid", data=mu_dense)

        for cname, (mean, phis, _) in components.items():
            grp = f.create_group(f"pca/{cname}")
            grp.create_dataset("mean_profile", data=mean.astype(np.float32))
            grp.create_dataset("eigenprofiles", data=phis.astype(np.float32))
            grp.create_dataset("pca_coeff_grid", data=pca_coeff_grids[cname])

        d = f.create_group("distributions")
        for name, alpha, zeta, omega, med in [
            ("epsilon1", e1_alpha, e1_zeta, e1_omega, e1_med),
            ("epsilon2", e2_alpha, e2_zeta, e2_omega, e2_med),
        ]:
            g = d.create_group(name)
            g.create_dataset("alpha", data=alpha)
            g.create_dataset("zeta", data=zeta)
            g.create_dataset("omega", data=omega)
            g.create_dataset("median", data=med)

    print("Export completed successfully.")
    return output_path


if __name__ == "__main__":
    export_params()
