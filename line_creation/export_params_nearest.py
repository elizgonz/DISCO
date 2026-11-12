"""
export_disco_params.py
──────────────────────────────────────────────────────────────────────────────
Exports DISCO parameters to HDF5 for the GRASS-E GPU pipeline.

What this script does
─────────────────────
1. Instantiates LineProfileGenerator(line), which loads pre-trained .npy
   files from DISCO/line_creation/data/line/.

2. Evaluates PCA splines and filling-factor distributions only at the requested
   mu samples (default: 1.0, 0.9, ..., 0.2).

3. Extracts DISCO's filling-factor distribution parameters (alpha, zeta, omega)
   and evaluates the exact median (0.5 quantile / 50th percentile) filling factor
   at each selected sample.

4. Writes the sparse, potentially nonuniform mu grid to HDF5. The paired Julia
   module uses the nearest sampled mu rather than interpolating between samples.
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import numpy as np
import h5py
from scipy.stats import skewnorm as scipy_skewnorm

from make_lines import LineProfileGenerator, exponential

LINE = "Fe6173"
MAX_DEG = 83.0
MU_MIN = float(np.nextafter(
    np.float32(np.cos(np.deg2rad(MAX_DEG))), np.float32(1.0)))
MU_MAX = 1.00
DEFAULT_MU_SAMPLES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)

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

def export_params(
    output_path=None, generator=None, line=LINE, mu_samples=DEFAULT_MU_SAMPLES
):
    """Export DISCO parameters at selected mu samples for nearest-neighbor use."""
    if generator is not None and generator.line != line:
        raise ValueError(
            f"Generator line {generator.line!r} does not match requested line {line!r}")
    print(f"Loading LineProfileGenerator for {line}...")
    gen = generator if generator is not None else LineProfileGenerator(line)
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "data", line,
            f"disco_{line}_params_nearest.h5")

    mu_samples = np.asarray(mu_samples, dtype=np.float32)
    if mu_samples.ndim != 1 or len(mu_samples) < 3:
        raise ValueError("Provide at least three mu samples for the quadratic CB fit")
    if not np.all(np.isfinite(mu_samples)):
        raise ValueError("mu samples must all be finite")
    if np.any(mu_samples < MU_MIN) or np.any(mu_samples > MU_MAX):
        raise ValueError(
            f"mu samples must be within DISCO's range [{MU_MIN}, {MU_MAX}]")
    mu_grid = np.unique(mu_samples)
    if len(mu_grid) != len(mu_samples):
        raise ValueError("mu samples must not contain duplicates")
    mu_grid.sort()
    deg_grid = np.minimum(
        np.degrees(np.arccos(mu_grid.astype(np.float64))), MAX_DEG)

    components = {
        "GT": (gen.mean_profile_gt, gen.eigenprofiles_gt, gen.coef_interp_gt),
        "OGR": (gen.mean_profile_ogr, gen.eigenprofiles_ogr, gen.coef_interp_ogr),
        "IgL": (gen.mean_profile_igl, gen.eigenprofiles_igl, gen.coef_interp_igl),
    }

    print(f"Evaluating DISCO parameters at {len(mu_grid)} selected mu samples...")
    pca_coeff_grids = {
        cname: np.stack([[interp(d) for d in deg_grid]
                         for interp in coef_interp]).astype(np.float32)
        for cname, (_, _, coef_interp) in components.items()
    }

    n_wave = len(gen.wl)
    n_pca = gen.eigenprofiles_gt.shape[0]

    print("Evaluating filling factor distribution parameters...")
    e1_alpha, e1_zeta, e1_omega, e1_med = get_distribution_params(
        gen.gt_med_coeffs, gen.gt_std_coeffs, gen.gt_skew_coeffs,
        deg_grid, is_gt=True)
    e2_alpha, e2_zeta, e2_omega, e2_med = get_distribution_params(
        gen.ratio_med_coeffs, gen.ratio_std_coeffs, gen.ratio_skew_coeffs,
        deg_grid, is_gt=False)

    print(f"Writing {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.attrs["line"] = line
        f.attrs["mu_min"] = float(mu_grid[0])
        f.attrs["mu_max"] = float(mu_grid[-1])
        f.attrs["n_mu"] = len(mu_grid)
        f.attrs["n_wave"] = n_wave
        f.attrs["n_pca"] = n_pca

        f.create_dataset("wavelength", data=gen.wl.astype(np.float32))
        f.create_dataset("mu_grid", data=mu_grid)

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
    parser = argparse.ArgumentParser(
        description="Export DISCO parameters at selected mu values for nearest-neighbor synthesis."
    )
    parser.add_argument("--line", default=LINE, choices=("Fe5250", "Fe6152", "Fe6173"))
    parser.add_argument(
        "--mu", type=float, nargs="+", default=DEFAULT_MU_SAMPLES,
        help="Known mu samples (default: 1.0 0.9 ... 0.2).",
    )
    parser.add_argument("--output", default=None, help="Output HDF5 path.")
    args = parser.parse_args()
    export_params(output_path=args.output, line=args.line, mu_samples=args.mu)
