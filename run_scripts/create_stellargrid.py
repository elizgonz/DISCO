import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from astropy.io import ascii as asc
from astropy.table import Table
from scipy.interpolate import splev, splrep
import datetime
import os
import argparse
import yaml
from shutil import copyfile
from scipy.spatial import ConvexHull
from astropy.io import fits
import shutil

base_dir = '/home/astro/phrrdx/stellar_absorption_lines/' #change this to your base directory
sys.path.append(base_dir + 'stellargrid_src/')
import stellargrid

config_name = 'solar_inc90_test'

################################################################
################################################################
"""Read in the config file."""
################################################################
################################################################
parser = argparse.ArgumentParser(description='Read the user inputs.')
parser.add_argument('-wdir','--work_dir', help = "Path to the working directory.",
                    type=str, required=False)
parser.add_argument('-cfg','--config_file_path', help = "Path to the croc_config.yaml.",
                    type=str, required=False)

args = vars(parser.parse_args())

config_file_path = f'{base_dir}/config_files/{config_name}.yaml'
with open(config_file_path) as f:
    config_dd = yaml.load(f,Loader=yaml.FullLoader)

savedir = f'{base_dir}/grids/{config_name}/'

"""Create the directory to save results."""
try:
    os.makedirs(savedir)
except OSError:
    savedir = savedir

print('Saving files in directory: ', savedir)

# ### Save the config file in the savedir 
shutil.copyfile(config_file_path, savedir + 'gridster_config.yaml')

# ################################################################
# ################################################################
# """Define output sub-directories."""
# ################################################################
# ################################################################

output_plots_dir = savedir + '/output_plots/'
try:
    os.makedirs(savedir + '/output_plots/')
except OSError:
    output_plots_dir = savedir + '/output_plots/'


output_data_dir = savedir + '/output_data/'
try:
    os.makedirs(savedir + '/output_data/')
except OSError:
    output_data_dir = savedir + '/output_data/'

# ###################################################################################### 
# ############ Initiate the grid  ######################
# ######################################################################################

star_dict = config_dd['star']

grid = stellargrid.StellarGrid(star_dict = star_dict)

df, dd = grid.get_stellar_tile_grid_snapshot()
df.to_csv(output_data_dir + 'stellar_grid.csv')


