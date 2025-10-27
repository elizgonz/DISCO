import sys
import os
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count

sys.path.append('line_creation')
from make_lines import LineProfileGenerator

os.nice(19)


def shift_profile(wl, I, vel):
    c = 299792458  # speed of light in m/s
    wl_shifted = wl * (1 + vel / c)
    interp_func = interp1d(wl_shifted, I, kind='linear', bounds_error=False, fill_value='extrapolate')
    I_shifted = interp_func(wl)
    return I_shifted


def process_tile(i, tile, wl, GT_quantile, ratio_quantile):
    deg = tile['angle']
    if deg > 83:
        deg = 83

    profile, _ = lpg.get_full_profile(deg, GT_quantile, ratio_quantile)
    og_I = profile
    proj_area = tile['proj_area']
    I_pA = og_I * proj_area
    vel = tile['los_vel']
    I_shifted = shift_profile(wl, I_pA, vel)
    return I_shifted


def run_disk_integration(working_folder, wl, iteration):
    tile_info = pd.read_csv(f'grids/{working_folder}/output_data/stellar_grid_visible.csv')
    GT_quantiles = pd.read_csv(f'grids/{working_folder}/output_data/GT_quantiles.csv')
    ratio_quantiles = pd.read_csv(f'grids/{working_folder}/output_data/ratio_quantiles.csv')

    GTs = GT_quantiles[f'GT_quantile_{iteration}']
    ratios = ratio_quantiles[f'ratio_quantile_{iteration}']

    all_Is = []
    for i, tile in tile_info.iterrows():
        I_shifted = process_tile(i, tile, wl, GTs[i], ratios[i])
        all_Is.append(I_shifted)
    all_Is = np.array(all_Is)

    final_profile = np.sum(all_Is, axis=0)

    return final_profile


def run_instance(args):
    working_folder, wl, i = args
    print(f'Processing iteration {i + 1}')
    return run_disk_integration(working_folder, wl, i)


def create_quantile_grid(working_folder, n_instances):
    tile_info_path = f'grids/{working_folder}/output_data/stellar_grid.csv'
    if not os.path.exists(tile_info_path):
        sys.exit(f"Error: grids/{working_folder} not found or incomplete. Please run create_stellar_grid first.")

    tile_info = pd.read_csv(tile_info_path)
    tile_info_visible = tile_info[tile_info['proj_area'] > 0]
    tile_info_visible = tile_info_visible[~tile_info_visible['proj_area'].isna()]

    tile_info_visible.to_csv(f'grids/{working_folder}/output_data/stellar_grid_visible.csv')

    GT_quantiles = np.random.uniform(0, 1, (len(tile_info_visible), n_instances))
    ratio_quantiles = np.random.uniform(0, 1, (len(tile_info_visible), n_instances))

    df_GT = pd.DataFrame(GT_quantiles, columns=[f'GT_quantile_{i}' for i in range(n_instances)])
    df_ratio = pd.DataFrame(ratio_quantiles, columns=[f'ratio_quantile_{i}' for i in range(n_instances)])

    df_GT.to_csv(f'grids/{working_folder}/output_data/GT_quantiles.csv')
    df_ratio.to_csv(f'grids/{working_folder}/output_data/ratio_quantiles.csv')


def main():
    parser = argparse.ArgumentParser(description='Run disk integration for a given grid and spectral line.')
    parser.add_argument('-g', '--grid_name', type=str, required=True,
                        help='Name of the grid folder inside "grids/".')
    parser.add_argument('-l', '--line', type=str, required=True,
                        help='Spectral line name (options: Fe6173, Fe5250, Fe6271, Fe6152).')
    parser.add_argument('-n', '--n_instances', type=int, default=1000,
                        help='Number of disk integration iterations.')
    parser.add_argument('-p', '--processes', type=int, default=cpu_count(),
                        help='Number of parallel processes (default: all available cores).')

    args = parser.parse_args()
    working_folder = args.grid_name
    line = args.line
    n_instances = args.n_instances
    n_processes = args.processes

    # Check grid folder exists
    grid_path = f'grids/{working_folder}'
    if not os.path.exists(grid_path):
        sys.exit(f"Error: grids/{working_folder} not found. \n Please create a config file named {working_folder} and run 'python run_scripts/create_stellar_grid -cn {working_folder}' first.")

    if line not in ['Fe6173', 'Fe5250', 'Fe6271', 'Fe6152']:
        sys.exit("Error: Invalid line name. Options are: Fe6173, Fe5250, Fe6271, Fe6152.")

    # Load wavelength data
    wl_path = f'line_creation/data/{line}/wl_air.txt'
    wl = np.loadtxt(wl_path)

    global lpg
    lpg = LineProfileGenerator(line)

    create_quantile_grid(working_folder, n_instances)

    args_list = [(working_folder, wl, i) for i in range(n_instances)]

    print(f"\nRunning with {n_instances} instances using {n_processes} processes...\n")

    with Pool(processes=n_processes) as pool:
        all_profs = pool.map(run_instance, args_list)

    all_profs = np.array(all_profs)

    save_dir = f'grids/{working_folder}/output_data/'
    np.save(f'{save_dir}{line}_DI_{n_instances}.npy', all_profs)

    print(f"\nDisk integration complete. Results saved to {save_dir}{line}_DI_{n_instances}.npy\n")


if __name__ == "__main__":
    main()
