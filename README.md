# DISCO - Disk Integrated Stellar Convection 

This repository contains a set of Python scripts to generate and analyse **disk-integrated stellar absorption line profiles**.  
The workflow involves creating a stellar surface grid, generating local spectral line profiles, and integrating them across the visible disk to simulate observed stellar spectra.

---

## Overview

The pipeline is divided into two main stages:

1. **Grid Creation (`run_scripts/create_stellar_grid.py`)**
   - Builds a stellar tile grid for a given configuration (e.g., solar-like star, specific inclination).
   - Saves geometric and physical parameters for each tile.

2. **Disk Integration (`run_scipts/disk_integrate.py`)**
   - Loads the pre-generated grid.
   - Simulates a given number of realizations of the disk-integrated spectral line.
   - Outputs an array of integrated line profiles that can be analysed statistically (e.g., velocity RMS).

---

## Instructions

1. **Create config file in config_files**
    - Create a yaml file in config_files 
    - Input the desired stellar parameters following the structure in the example config file - config_files/solar_inc90.yaml 

2. **Create the stellar grid**
    - From the base DISCO directory, run 
    
    ```bash
    python run_scripts/create_stellar_grid.py -cn config_name
    ```

    where config_name is the name of your yaml file 

    - A folder will be created in the grids directory named after your config file, this folder contains:
        - A copy of the config
        - An output_data folder containing grid parameters 
        - An empty output_plots folder

3. **(Optional) Create plots of your grid**
    - From the base DISCO directory, run 
    
    ```bash
    python run_scripts/make_grid_plot.py -gn grid_name
    ```

    where grid_name is the name of your grid folder (same as config_name)

    - A pdf plot will be stored in grids/grid_name/output_plots/grid_plot.pdf showing your grid setup
    - Note if the number of tiles on your grid is large this can be slow 

4. **Simulate disk integrated absorption lines**

    - From the base DISCO directory, run 
    
    ```bash
    python run_scripts/disk_integrate.py -g grid_name -l line -n 100 -p 50 
    ```

    where the inputs are the following:
        -g: (required) name of the grid folder 
        -l: (required) line you are working with (options: Fe6173, Fe5250, Fe6271, Fe6152)
        -n: (otpional) number of instances to calculate (defaults to 100)
        -p: number of parallel processes to use (defaults to available cores)

    - Results will be stored in grid_name/output_data 










