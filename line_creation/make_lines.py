import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import skewnorm
from matplotlib import cm
from matplotlib.colors import ListedColormap
import line_analysis as ba

# Get the 'inferno' colormap
inferno = cm.get_cmap('inferno', 256)

# Remove the top 10% (brightest colors)
newcolors = inferno(np.linspace(0, 0.8, 256))  
custom_inferno = ListedColormap(newcolors)

dp_color = inferno(0.1)
np_color = inferno(0.4)
bp_color = inferno(0.75)

# Set font to 'Times New Roman' for academic journal style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # Use 'Palatino' or other serif fonts if preferred
    'font.size': 15,                    # Adjust font size as needed (e.g., 12-14 for publications)
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# summed exponential function
def exponential(x, a, a2, b, b2, c):
    return (a * np.exp(b * x)) + (a2 * np.exp(b2 * x)) + c

class LineProfileGenerator:
    def __init__(self, line):
        self.line = line
        base_path = f'/home/astro/phrrdx/stellar_absorption_lines/line_creation/data/{line}'

        self.mean_profile_gt = np.load(f'{base_path}/component_profiles/mean_profile_gt.npy')
        self.mean_profile_ogr = np.load(f'{base_path}/component_profiles/mean_profile_ogr.npy')
        self.mean_profile_igl = np.load(f'{base_path}/component_profiles/mean_profile_igl.npy')

        self.eigenprofiles_gt = np.load(f'{base_path}/component_profiles/eigenprofiles_gt.npy')
        self.eigenprofiles_ogr = np.load(f'{base_path}/component_profiles/eigenprofiles_ogr.npy')
        self.eigenprofiles_igl = np.load(f'{base_path}/component_profiles/eigenprofiles_igl.npy')

        self.coef_interp_gt = np.load(f'{base_path}/component_profiles/coef_interp_gt.npy', allow_pickle=True)
        self.coef_interp_ogr = np.load(f'{base_path}/component_profiles/coef_interp_ogr.npy', allow_pickle=True)
        self.coef_interp_igl = np.load(f'{base_path}/component_profiles/coef_interp_igl.npy', allow_pickle=True)

        self.gt_med_coeffs = np.load(f'{base_path}/filling_factors/GT_median.npy')
        self.gt_std_coeffs = np.load(f'{base_path}/filling_factors/GT_std_norm.npy')
        self.gt_skew_coeffs = np.load(f'{base_path}/filling_factors/GT_skew.npy')

        self.ratio_med_coeffs = np.load(f'{base_path}/filling_factors/ratio_median.npy')
        self.ratio_std_coeffs = np.load(f'{base_path}/filling_factors/ratio_std.npy')
        self.ratio_skew_coeffs = np.load(f'{base_path}/filling_factors/ratio_skew.npy')
        
        self.wl = np.loadtxt(f'{base_path}/wl_air.txt')

    def reconstruct_profile(self, deg, coef_interp, eigenprofiles, mean_profile):
        a = np.array([interp(deg) for interp in coef_interp])
        return mean_profile + a.dot(eigenprofiles)

    def generate_ffs(self, deg, GT_quantile, ratio_quantile):
        poly_med = np.poly1d(self.gt_med_coeffs)
        gt_med_val = poly_med(deg)
        gt_std_val = exponential(deg, *self.gt_std_coeffs) * gt_med_val
        poly_skew = np.poly1d(self.gt_skew_coeffs)
        gt_skew_val = poly_skew(deg)

        gt_dist = skewnorm(a=gt_skew_val, loc=gt_med_val, scale=gt_std_val)
        gt_ff = gt_dist.ppf(GT_quantile)

        poly_med = np.poly1d(self.ratio_med_coeffs)
        ratio_med_val = poly_med(deg)
        ratio_std_val = exponential(deg, *self.ratio_std_coeffs)
        poly_skew = np.poly1d(self.ratio_skew_coeffs)
        ratio_skew_val = poly_skew(deg)

        ratio_dist = skewnorm(a=ratio_skew_val, loc=ratio_med_val, scale=ratio_std_val)
        ratio_ff = ratio_dist.ppf(ratio_quantile)
        if ratio_ff < 0:
            print(f'Ratio FF < 0 (GT: {GT_quantile}, Ratio: {ratio_quantile}, Deg: {deg}): {ratio_ff}')

        igl_ff = (1 - gt_ff) / (1 + ratio_ff)
        ogr_ff = ratio_ff * igl_ff

        return gt_ff, igl_ff, ogr_ff

    def get_full_profile(self, deg, GT_quantile, ratio_quantile):

        if deg > 83:
            raise ValueError(f'Degree must be <= 83 to avoid extrapolation issues. Given: {deg}')

        gt_comp = self.reconstruct_profile(deg, self.coef_interp_gt, self.eigenprofiles_gt, self.mean_profile_gt)
        igl_comp = self.reconstruct_profile(deg, self.coef_interp_igl, self.eigenprofiles_igl, self.mean_profile_igl)
        ogr_comp = self.reconstruct_profile(deg, self.coef_interp_ogr, self.eigenprofiles_ogr, self.mean_profile_ogr)

        gt_ff, igl_ff, ogr_ff = self.generate_ffs(deg, GT_quantile, ratio_quantile)
        ffs = [gt_ff, igl_ff, ogr_ff]
        full_prof = gt_comp * gt_ff + igl_comp * igl_ff + ogr_comp * ogr_ff

        #error message if nan in profile
        if np.any(np.isnan(full_prof)):
            print(f'get_full_profile: NaN in profile (GT: {GT_quantile}, Ratio: {ratio_quantile}, Deg: {deg})')

        return full_prof, ffs
    
    def make_test_plot(self, deg, n_instances):

        if deg > 83:
            print(f'Degree must be <= 83 to avoid extrapolation issues. Given: {deg}')
            return

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        wl = self.wl

        GT_quantiles = np.random.uniform(0, 1, n_instances)
        ratio_quantiles = np.random.uniform(0, 1, n_instances)

        full_profs = np.zeros((n_instances, len(wl)))
        gt_ffs = np.zeros(n_instances)
        igl_ffs = np.zeros(n_instances)
        ogr_ffs = np.zeros(n_instances)


        for i in range(n_instances):
            gt_quantile = GT_quantiles[i]
            ratio_quantile = ratio_quantiles[i]
            
            # Generate the line profile
            prof, ffs = self.get_full_profile(deg, gt_quantile, ratio_quantile)

            full_profs[i] = prof

            #check for nans
            if np.any(np.isnan(prof)):
                print(f'NaN in profile for instance {i} (GT: {gt_quantile}, Ratio: {ratio_quantile})')

            gt_ffs[i] = ffs[0]
            igl_ffs[i] = ffs[1]
            ogr_ffs[i] = ffs[2]

        #plot the three components
        gt_comp = self.reconstruct_profile(deg, self.coef_interp_gt, self.eigenprofiles_gt, self.mean_profile_gt)
        igl_comp = self.reconstruct_profile(deg, self.coef_interp_igl, self.eigenprofiles_igl, self.mean_profile_igl)
        ogr_comp = self.reconstruct_profile(deg, self.coef_interp_ogr, self.eigenprofiles_ogr, self.mean_profile_ogr)
        axs[0].plot(wl, gt_comp, label='GT', color=bp_color)
        axs[0].plot(wl, igl_comp, label='IgL', color=dp_color)
        axs[0].plot(wl, ogr_comp, label='OGR', color=np_color)
        axs[0].set_title('Components')
        axs[0].set_xlabel('Wavelength (nm)')
        axs[0].set_ylabel('Intensity')
        axs[0].set_xlim(wl[50], wl[-50])

        axs[1].hist(gt_ffs, bins=20, color=bp_color, alpha=0.5, label='GT')
        axs[1].hist(igl_ffs, bins=20, color=dp_color, alpha=0.5, label='IgL')
        axs[1].hist(ogr_ffs, bins=20, color=np_color, alpha=0.5, label='OGR')
        axs[1].set_title('Filling Factors')
        axs[1].set_xlabel('Filling Factor')
        axs[1].set_ylabel('Count')
        axs[1].legend()

        #plot the full profiles
        for i in range(n_instances):
            bisx, bisy = ba.calculate_line_bisector(wl, full_profs[i])
            axs[2].plot(bisx, bisy, color='blue', alpha=0.5)

        axs[2].set_title('Bisectors')
        axs[2].set_xlabel('Wavelength (nm)')
        axs[2].set_ylabel('Intensity')

        #big title
        fig.suptitle(f'Line Creation: {self.line}, {n_instances} Instances, Degree {deg}', fontsize=20)

