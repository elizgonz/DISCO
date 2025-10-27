import sys
sys.path.append('/home/astro/phrrdx/stellar_absorption_lines/line_creation')
from make_lines import LineProfileGenerator
sys.path.append('/home/astro/phrrdx/generate_new_profiles/line_creation/import_scripts')
import get_data as gd
import line_profile_analysis as lpa


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from multiprocessing import Pool
import os
os.nice(19)

base_dir = '/home/astro/phrrdx/stellar_absorption_lines/' #CHANGE THIS TO YOUR BASE DIRECTORY

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

def run_disk_integration(working_folder, wl, iteration, save_tile_results):
    tile_info = pd.read_csv(f'{base_dir}/grids/{working_folder}/output_data/stellar_grid_visible.csv')

    GT_quantiles = pd.read_csv(f'{base_dir}/grids/{working_folder}/output_data/GT_quantiles.csv')
    ratio_quantiles = pd.read_csv(f'{base_dir}/grids/{working_folder}/output_data/ratio_quantiles.csv')

    GTs = GT_quantiles[f'GT_quantile_{iteration}']
    ratios = ratio_quantiles[f'ratio_quantile_{iteration}']

    all_Is = []
    for i, tile in tile_info.iterrows():
        I_shifted = process_tile(i, tile, wl, GTs[i], ratios[i])
        all_Is.append(I_shifted)
    all_Is = np.array(all_Is)
    
    final_profile = np.sum(all_Is, axis=0)

    template, _ = lpg.get_full_profile(0, 0.5, 0.5)

    print(all_Is.shape, all_Is[0].shape, template.shape)

    if save_tile_results:

        RVs = [np.real(lpa.get_rv(wl, np.array(all_Is[i]), template)) for i in range(len(all_Is))]
        continuums = [np.max(all_Is[i]) for i in range(len(all_Is))]
        
        #create dataframe for RVs and continuums
        results_df = pd.DataFrame(index=range(len(tile_info)), 
                                columns=[f'RV_{iteration}', f'continuum_{iteration}'])
        results_df[f'RV_{iteration}'] = RVs
        results_df[f'continuum_{iteration}'] = continuums

        results_df.to_csv(f'{base_dir}/grids/{working_folder}/output_data/results_iteration{iteration}.csv')

    return final_profile

def run_instance(args):
    # Unpack arguments for multiprocessing pool map
    working_folder, wl, i, save_tile_results = args
    print(f'Processing iteration {i+1} / {n_instances}...')
    return run_disk_integration(working_folder, wl, i, save_tile_results)

def create_quantile_grid(working_folder, n_instances):

    tile_info = pd.read_csv(f'{base_dir}/grids/{working_folder}/output_data/stellar_grid.csv')

    tile_info_visible = tile_info[tile_info['proj_area'] > 0]
    tile_info_visible = tile_info_visible[~tile_info_visible['proj_area'].isna()]

    tile_info_visible.to_csv(f'{base_dir}/grids/{working_folder}/output_data/stellar_grid_visible.csv')

    GT_quantiles = np.zeros((len(tile_info_visible), n_instances))
    ratio_quantiles = np.zeros((len(tile_info_visible), n_instances))

    for i in range(n_instances):
        GT_quantiles[:, i] = np.random.uniform(0, 1, len(tile_info_visible))
        ratio_quantiles[:, i] = np.random.uniform(0, 1, len(tile_info_visible))

    df_GT = pd.DataFrame(GT_quantiles, columns=[f'GT_quantile_{i}' for i in range(n_instances)])
    df_ratio = pd.DataFrame(ratio_quantiles, columns=[f'ratio_quantile_{i}' for i in range(n_instances)])

    df_GT.to_csv(f'{base_dir}/grids/{working_folder}/output_data/GT_quantiles.csv')
    df_ratio.to_csv(f'{base_dir}/grids/{working_folder}/output_data/ratio_quantiles.csv')

def main(save_tile_results = False):
    working_folder = 'solar_inc90'
    line = 'Fe6271'
    wl = gd.get_wl('HD', line)
    global n_instances
    n_instances = 1000

    global lpg 
    lpg = LineProfileGenerator(line)

    create_quantile_grid(working_folder, n_instances)
    
    # Prepare arguments list for each instance
    args_list = [(working_folder, wl, i, save_tile_results) for i in range(n_instances)]
    
    # Use multiprocessing over n_instances
    with Pool(processes=100) as pool:  # set processes to number of CPU cores or desired count
        all_profs = pool.map(run_instance, args_list)
    
    all_profs = np.array(all_profs)
    
    save_dir = f'{base_dir}/grids/{working_folder}/output_data/'
    np.save(f'{save_dir}{line}_DI_{n_instances}_train.npy', all_profs)

    if save_tile_results:

        #combine the results csvs
        results_dfs = []
        for i in range(n_instances):
            results_df = pd.read_csv(f'{save_dir}results_iteration{i}.csv', index_col=0)
            results_dfs.append(results_df)

        combined_results = pd.concat(results_dfs, axis=1)
        combined_results.to_csv(f'{save_dir}results_combined_{line}.csv')

        #remove individual results csvs
        for i in range(n_instances):
            os.remove(f'{save_dir}results_iteration{i}.csv')

if __name__ == "__main__":
    main(save_tile_results=False)