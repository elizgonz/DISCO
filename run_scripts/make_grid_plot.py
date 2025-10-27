import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy.spatial import ConvexHull
import matplotlib.cm as cm
from matplotlib.colors import Normalize

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

parser = argparse.ArgumentParser(description='Read the user inputs.')

parser.add_argument(
    '-gn', '--grid_name',
    help="Name of the grid folder in grids/.",
    type=str, required=True
)

args = vars(parser.parse_args())

config_name = args['grid_name']

working_folder = f'grids/{config_name}/'
savedir = f'{working_folder}/output_plots/'
df = pd.read_csv(f'{working_folder}/output_data/stellar_grid.csv')

#only include visible tiles
df = df[df['proj_area'] > 0]
#reset index
df.reset_index(drop=True, inplace=True)


fig, axs = plt.subplots(1, 3, figsize=(24, 7))  # Wide figure for 3 plots

# --- Plot 1: Velocity Grid ---
los_vel = df['los_vel']
cmap_vel = cm.RdBu_r
norm_vel = Normalize(vmin=np.min(los_vel), vmax=np.max(los_vel))

for it in range(len(df)):
    points = np.array([df['x1'][it], df['y1'][it],
                        df['x2'][it], df['y2'][it],
                        df['x3'][it], df['y3'][it],
                        df['x4'][it], df['y4'][it]]).reshape(4, 2)
    
    hull = ConvexHull(points, qhull_options='QJ')
    color = cmap_vel(norm_vel(los_vel[it]))

    axs[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color)
    vertices = np.append(hull.vertices, hull.vertices[0])
    axs[0].plot(points[vertices, 0], points[vertices, 1], 'k-', linewidth=0.01)

axs[0].axis('scaled')
axs[0].set_title('Rotational Velocity')
sm_vel = plt.cm.ScalarMappable(cmap=cmap_vel, norm=norm_vel)
sm_vel.set_array([])
cbar_vel = fig.colorbar(sm_vel, ax=axs[0])
cbar_vel.set_label('Line-of-sight velocity (m/s)')

print(f"Min velocity: {np.min(los_vel)} m/s, Max velocity: {np.max(los_vel)} m/s")

# --- Plot 2: Tile Areas ---
cmap_area = cm.viridis
norm_area = Normalize(vmin=np.min(df['proj_area']), vmax=np.max(df['proj_area']))

for it in range(len(df)):
    points = np.array([df['x1'][it], df['y1'][it],
                        df['x2'][it], df['y2'][it],
                        df['x3'][it], df['y3'][it],
                        df['x4'][it], df['y4'][it]]).reshape(4, 2)
    hull = ConvexHull(points, qhull_options='QJ')
    color = cmap_area(norm_area(df['proj_area'][it]))
    axs[1].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color)
    vertices = np.append(hull.vertices, hull.vertices[0])
    axs[1].plot(points[vertices, 0], points[vertices, 1], 'k-', linewidth=0.01)

axs[1].axis('scaled')
axs[1].set_title('Tile Areas')
sm_area = plt.cm.ScalarMappable(cmap=cmap_area, norm=norm_area)
sm_area.set_array([])
cbar_area = fig.colorbar(sm_area, ax=axs[1])
cbar_area.set_label('Projected Area (Mm²)')

print(f"Min area: {np.min(df['proj_area'])} Mm², Max area: {np.max(df['proj_area'])} Mm²")

# --- Plot 3: Viewing Angle ---
cmap_angle = cm.viridis
norm_angle = Normalize(vmin=np.min(df['angle']), vmax=90)

for it in range(len(df)):
    points = np.array([df['x1'][it], df['y1'][it],
                        df['x2'][it], df['y2'][it],
                        df['x3'][it], df['y3'][it],
                        df['x4'][it], df['y4'][it]]).reshape(4, 2)
    hull = ConvexHull(points, qhull_options='QJ')
    color = cmap_angle(norm_angle(df['angle'][it]))
    axs[2].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color)
    vertices = np.append(hull.vertices, hull.vertices[0])
    axs[2].plot(points[vertices, 0], points[vertices, 1], 'k-', linewidth=0.01)

axs[2].axis('scaled')
axs[2].set_title('Viewing Angle')
sm_angle = plt.cm.ScalarMappable(cmap=cmap_angle, norm=norm_angle)
sm_angle.set_array([])
cbar_angle = fig.colorbar(sm_angle, ax=axs[2])
cbar_angle.set_label(r'Viewing angle ($^\circ$)')

print(f"Min angle: {np.min(df['angle'])} degrees, Max angle: {np.max(df['angle'])} degrees")

plt.tight_layout()
plt.savefig(f'{savedir}/grid_plot.pdf')
plt.close()

