import sys
import os
import argparse
import yaml
import shutil
sys.path.append('stellargrid_src/')
import stellargrid


# ###################################################################################### 
# Read in the config file.
# ###################################################################################### 

parser = argparse.ArgumentParser(description='Read the user inputs.')

parser.add_argument(
    '-cn', '--config_name',
    help="Name of the config file (without .yaml extension).",
    type=str, required=True
)

args = vars(parser.parse_args())

config_name = args['config_name']
config_file_path = f'config_files/{config_name}.yaml'

with open(config_file_path) as f:
    config_dd = yaml.load(f, Loader=yaml.FullLoader)

savedir = f'grids/{config_name}/'

"""Create the directory to save results."""
os.makedirs(savedir, exist_ok=True)
print('Saving files in directory: ', savedir)

# ### Save the config file in the savedir 
shutil.copyfile(config_file_path, savedir + 'gridster_config.yaml')

# ###################################################################################### 
# Define output sub-directories.
# ###################################################################################### 

output_plots_dir = os.path.join(savedir, 'output_plots/')
os.makedirs(output_plots_dir, exist_ok=True)

output_data_dir = os.path.join(savedir, 'output_data/')
os.makedirs(output_data_dir, exist_ok=True)

# ###################################################################################### 
# Initiate the grid  
# ######################################################################################

star_dict = config_dd['star']

grid = stellargrid.StellarGrid(star_dict=star_dict)

df, dd = grid.get_stellar_tile_grid_snapshot()
df.to_csv(output_data_dir + 'stellar_grid.csv')
