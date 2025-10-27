import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
from multiprocessing import Pool, cpu_count

import pandas as pd
import sys
sys.path.append('/home/astro/phrrdx/generate_new_profiles/line_creation')
sys.path.append('/home/astro/phrrdx/generate_new_profiles/line_creation/import_scripts')
import get_data as gd
import line_profile_analysis as lpa
import make_lines
import bisector_analysis as ba
import os 

import pickle

os.nice(19)

# -------------------
# Logging setup
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------
# BIS calculation
# -------------------
def get_BIS(bisector_wavelengths, bisector_fluxes, lower=[0.2, 0.51], upper=[0.61, 0.83]):
    bisector_fluxes = (bisector_fluxes - np.min(bisector_fluxes)) / (np.max(bisector_fluxes) - np.min(bisector_fluxes))

    top_range = (bisector_fluxes > upper[0]) & (bisector_fluxes < upper[1])
    bottom_range = (bisector_fluxes > lower[0]) & (bisector_fluxes < lower[1])

    if not np.any(top_range) or not np.any(bottom_range):
        return np.nan

    return np.mean(bisector_wavelengths[top_range]) - np.mean(bisector_wavelengths[bottom_range])

# -------------------
# Worker for multiprocessing
# -------------------
def evaluate_bounds(bounds):
    low_start, low_end, up_start, up_end, bisector_wavelengths_list, bisector_fluxes_list, rv_ccfs = bounds

    bis_values = []
    for wl, fl in zip(bisector_wavelengths_list, bisector_fluxes_list):
        bis = get_BIS(wl, fl, lower=[low_start, low_end], upper=[up_start, up_end])
        if not np.isnan(bis):
            bis_values.append(bis)
    bis_values = np.array(bis_values)

    if len(bis_values) < 5:
        return None  # invalid case

    corr, _ = pearsonr(bis_values, rv_ccfs[:len(bis_values)])
    return corr, (low_start, low_end), (up_start, up_end), bis_values

# -------------------
# Optimisation loop (parallel)
# -------------------
def optimise_BIS(bisector_wavelengths_list, bisector_fluxes_list, rv_ccfs):
    depth_grid = np.arange(0.05, 0.96, 0.05)

    tasks = []
    for low_start in depth_grid:
        for low_end in depth_grid:
            if low_end <= low_start + 0.05:
                continue
            for up_start in depth_grid:
                for up_end in depth_grid:
                    if up_end <= up_start + 0.05:
                        continue
                    if low_end >= up_start:
                        continue
                    tasks.append((low_start, low_end, up_start, up_end,
                                  bisector_wavelengths_list, bisector_fluxes_list, rv_ccfs))

    best_corr = 0
    best_bounds = None
    best_bis_values = None

    with Pool(processes=min(50, cpu_count())) as pool:
        for result in pool.imap_unordered(evaluate_bounds, tasks, chunksize=50):
            if result is None:
                continue
            corr, low, up, bis_values = result
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_bounds = (low, up)
                best_bis_values = bis_values
                logging.info(f"New best correlation: {best_corr:.3f} with bounds {best_bounds}")

    return best_corr, best_bounds, best_bis_values

# -------------------
# Scatter reduction calculation
# -------------------
def compute_scatter_reduction(bis_values, rv_vals):
    slope, intercept, _, _, _ = linregress(bis_values, rv_vals)
    rv_pred = slope * np.array(bis_values) + intercept
    residuals = rv_vals - rv_pred
    scatter_orig = np.std(rv_vals)
    scatter_resid = np.std(residuals)
    reduction = 1 - (scatter_resid / scatter_orig)
    return reduction

# -------------------
# Combined plotting
# -------------------
def plot_summary(results):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharey='row')
    lines = list(results.keys())

    plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # Use 'Palatino' or other serif fonts if preferred
    'font.size': 16,                    # Adjust font size as needed (e.g., 12-14 for publications)
    'axes.titlesize': 30,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 14,
    'figure.titlesize': 40
    })

    titles = {
        'Fe5250': 'Fe I 525.0 nm',
        'Fe6152': 'Fe I 615.2 nm',
        'Fe6173': 'Fe I 617.3 nm',
        'Fe6271': 'Fe I 627.1 nm'
    }

    # --- Get global BIS/RV ranges for consistent binning
    y_min = min(np.min(res["rv_ccfs"]) for res in results.values())
    y_max = max(np.max(res["rv_ccfs"]) for res in results.values())

    for i, line in enumerate(lines):
        res = results[line]
        lower, upper = res["best_bounds"]
        mean_flux = res["mean_flux"]
        mean_wl = res["mean_wl"]
        bis_values = res["best_bis_values"]
        rv_vals = res["rv_ccfs"][:len(bis_values)]

        # --- Top row: bisector plot
        ax1 = axs[0, i]
        ax1.plot(mean_wl, mean_flux, 'k-', lw=1.5, label="Mean bisector")
        ax1.axhspan(lower[0], lower[1], color="blue", alpha=0.3, label="Lower region")
        ax1.axhspan(upper[0], upper[1], color="deeppink", alpha=0.3, label="Upper region")
        ax1.set_title(titles[line])
        ax1.set_xlabel("Velocity (m/s)")
        axs[0,0].set_ylabel("Normalized Flux")
        ax1.legend()

        # --- Bottom row: BIS vs RV with fixed extent
        ax2 = axs[1, i]
        hb = ax2.hexbin(
            bis_values, rv_vals,
            gridsize=30, cmap="RdPu_r",
            extent=(min(bis_values), max(bis_values), y_min, y_max),
            vmin=0   # ensure empty bins show as 0
        )
        fig.colorbar(hb, ax=ax2, label="Counts")
        ax2.set_xlabel("BIS value")
        axs[1,0].set_ylabel("RV (m/s)")

        # Annotate correlation and scatter reduction
        R = res["best_corr"]
        scatter_red = res["scatter_reduction"] * 100
        ax2.text(
            0.10, 0.20,
            f"R = {R:.3f}\nScatter ↓ {scatter_red:.1f}%",
            transform=ax2.transAxes,
            fontsize=16,
            va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
        )

    plt.tight_layout()
    plt.savefig("BIS_RV_summary.pdf")
    plt.show()


# -------------------
# Main workflow
if __name__ == "__main__":
    results = {}

    bisxs = np.real(np.load('bisxs.npy'))
    bisys = np.real(np.load('bisys.npy'))
    rv_ccfs = np.real(np.load('rvs_og.npy'))

    best_corr, best_bounds, best_bis_values = optimise_BIS(bisxs, bisys, rv_ccfs)
    scatter_reduction = compute_scatter_reduction(best_bis_values, rv_ccfs[:len(best_bis_values)])

    all_fluxes, all_waves = [], []
    for wl_arr, fl_arr in zip(bisxs, bisys):
        norm_flux = (fl_arr - np.min(fl_arr)) / (np.max(fl_arr) - np.min(fl_arr))
        all_fluxes.append(norm_flux)
        all_waves.append(wl_arr)
    mean_flux = np.mean(all_fluxes, axis=0)
    mean_wl = np.mean(all_waves, axis=0)

    results['Fe6173'] = dict(
        best_corr=best_corr,
        best_bounds=best_bounds,
        best_bis_values=best_bis_values,
        rv_ccfs=rv_ccfs,
        scatter_reduction=scatter_reduction,
        mean_flux=mean_flux,
        mean_wl=mean_wl
    )

    logging.info(f"Best correlation {best_corr:.3f}, Scatter reduction {scatter_reduction*100:.1f}%")

    # --- Save results ---
    with open("BIS_results.pkl", "wb") as f:
        pickle.dump(results, f)

    logging.info("Results saved to BIS_results.pkl")
