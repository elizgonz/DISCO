import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.append('/home/astro/phrrdx/generate_new_profiles/gridster/')
import stellargrid
from shutil import copyfile
from scipy.spatial import ConvexHull
from astropy.io import fits
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
from matplotlib.colors import TwoSlopeNorm

# Set font to 'Times New Roman' for academic journal style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # Use 'Palatino' or other serif fonts if preferred
    'font.size': 16,                    # Adjust font size as needed (e.g., 12-14 for publications)
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

line = 'Fe6152'

working_folder = '/home/astro/phrrdx/stellar_absorption_lines/grids/solar_inc90/'
sg_file = f'{working_folder}/output_data/stellar_grid_visible.csv'
results_file = f'{working_folder}/output_data/results_combined_{line}.csv'
savedir = f'{working_folder}/output_plots/'

# Create save directory if it doesn't exist
os.makedirs(savedir, exist_ok=True)

sg_df = pd.read_csv(sg_file)
results_df = pd.read_csv(results_file, index_col=0)

def calculate_derived_quantities(df, num_iterations):
    """Calculate mean values, CB, and RMS for all quantities"""
    
    # Calculate means
    results_df['mean_rv'] = results_df[[f'RV_{i}' for i in range(num_iterations)]].mean(axis=1)
    results_df['mean_continuum_area'] = results_df[[f'continuum_{i}' for i in range(num_iterations)]].mean(axis=1)
    
    # Calculate convective blueshift (CB)
    results_df['CB'] = results_df['mean_rv'] - sg_df['los_vel']
    results_df['mean_continuum'] = results_df['mean_continuum_area'] / sg_df['proj_area']
    
    # Calculate RMS for RV
    rms_rvs = np.zeros(len(df))
    for i in range(len(df)):
        print(f"Calculating RMS for tile {i+1}/{len(df)}")
        rms_rvs[i] = np.sqrt(np.mean((results_df[[f'RV_{j}' for j in range(num_iterations)]].iloc[i] - results_df['mean_rv'][i])**2))
    results_df['rms_rv'] = rms_rvs

    new_derived_columns = ['mean_rv', 'mean_continuum', 'CB', 'rms_rv']
    #create new dataframe with only the derived columns
    derived_df = results_df[new_derived_columns].copy()

    #save as csv
    derived_df.to_csv(f'{savedir}/derived_quantities_{line}_100.csv', index=False)
    
    return results_df

def plot_grid_variable(variable_name, colorbar_label, filename, cmap_name='inferno'):
    """
    Generic function to plot any variable on the stellar grid
    
    :param variable_name: Column name in results_df to plot
    :param colorbar_label: Label for the colorbar
    :param filename: Output filename (without extension)
    :param cmap_name: Colormap name
    """
    
    # Create figure with defined axes - important for colorbar
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cmap = getattr(cm, cmap_name)
    variable = results_df[variable_name]
    #normaise to be between 0 and 1
    if variable_name == 'mean_continuum':
        variable = (variable - np.min(variable)) / (np.max(variable) - np.min(variable))

    norm = Normalize(vmin=np.min(variable), vmax=np.max(variable))
    
    for it in range(len(results_df)):
        points = np.array([sg_df['x1'][it], sg_df['y1'][it],
                        sg_df['x2'][it], sg_df['y2'][it],
                        sg_df['x3'][it], sg_df['y3'][it],
                        sg_df['x4'][it], sg_df['y4'][it]]).reshape(4, 2)
        hull = ConvexHull(points, qhull_options='QJ')
        
        color = cmap(norm(variable[it]))
        # ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'k-', linewidth=0.01)  # Draw edges
        ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color)
        
    ax.axis('scaled')
    ax.set_xlabel('X position (Mm)')
    ax.set_ylabel('Y position (Mm)')
    # ax.set_title(f'{colorbar_label} Map')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(colorbar_label, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{savedir}/{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{savedir}/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {filename} plot to {savedir}")

def plot_all_variables(num_iterations):
    """Plot all four variables: RV RMS, mean RV, CB, and continuum"""
    
    # First calculate all derived quantities
    global results_df
    results_df = calculate_derived_quantities(results_df, num_iterations)
    
    # Plot RV RMS
    plot_grid_variable('rms_rv', 'RV RMS (m/s)', f'{line}_rms_rv_plot', 'inferno')

    
    # Plot mean RV
    plot_grid_variable('mean_rv', 'Mean RV (m/s)', f'{line}_mean_rv_plot', 'RdBu_r')

    
    # Plot convective blueshift (CB)
    plot_grid_variable('CB', 'Convective Blueshift (m/s)', f'{line}_CB_plot', 'inferno_r')

    
    # Plot mean continuum
    plot_grid_variable('mean_continuum', 'Normalised Continuum Intensity', f'{line}_continuum_plot', 'plasma')

    
    # print("All plots saved successfully!")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"RV RMS range: {results_df['rms_rv'].min():.2f} to {results_df['rms_rv'].max():.2f} m/s")
    print(f"Mean RV range: {results_df['mean_rv'].min():.2f} to {results_df['mean_rv'].max():.2f} m/s")
    print(f"CB range: {results_df['CB'].min():.2f} to {results_df['CB'].max():.2f} m/s")
    print(f"Continuum range: {results_df['mean_continuum'].min():.4f} to {results_df['mean_continuum'].max():.4f}")

def plot_summary_1x3(results_df):
    """Create a 1x3 figure showing CB, RV RMS, and Continuum side by side"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    variable_list = ['CB', 'rms_rv', 'mean_continuum']
    labels = ['Granulation induced RV (m/s)', 'RV RMS (m/s)', 'Normalised Continuum Intensity']
    cmaps = ['RdBu', 'inferno', 'inferno']

    for ax, var, label, cmap_name in zip(axes, variable_list, labels, cmaps):
        cmap = getattr(cm, cmap_name)
        variable = results_df[var]

        # # Normalise continuum to 0–1
        if var == 'mean_continuum':
            variable = variable / np.max(variable)

        if var == 'CB':
            variable -= 410
            abs_max = np.max(np.abs(variable))
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        else:

            norm = Normalize(vmin=np.min(variable), vmax=np.max(variable))

        for it in range(len(results_df)):
            points = np.array([sg_df['x1'][it], sg_df['y1'][it],
                               sg_df['x2'][it], sg_df['y2'][it],
                               sg_df['x3'][it], sg_df['y3'][it],
                               sg_df['x4'][it], sg_df['y4'][it]]).reshape(4, 2)
            hull = ConvexHull(points, qhull_options='QJ')
            color = cmap(norm(variable[it]))
            ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color)

        ax.axis('scaled')
        ax.set_xlabel('X (Mm)')
        if ax == axes[0]:
            ax.set_ylabel('Y (Mm)')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label(label)

    plt.savefig(f'{savedir}/{line}_summary_1x3.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{savedir}/{line}_summary_1x3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 1x3 summary plot to {savedir}")


# Run the plotting function
# results_df = calculate_derived_quantities(results_df, 100)
results_df = pd.read_csv(f'{savedir}/derived_quantities_{line}_100.csv')
plot_summary_1x3(results_df)