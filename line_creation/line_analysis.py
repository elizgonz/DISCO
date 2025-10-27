import numpy as np
from scipy import interpolate

def interpolate_crossing(x, y, level):

    #The interpolation doesn't like duplicates - remove them 
    unique_y, unique_indices = np.unique(y, return_index=True)
    unique_x = x[unique_indices]

    # Check if the unique arrays have enough points for cubic interpolation
    if len(unique_y) < 4:
        # If not use linear interpolation
        f = interpolate.interp1d(unique_y, unique_x, kind='linear', fill_value='extrapolate')
    else:
        f = interpolate.interp1d(unique_y, unique_x, kind='cubic', fill_value='extrapolate')
    
    return f(level)


def calculate_line_bisector(og_wavelength, og_flux, num_levels=50, min_level=1.1, max_level=0.99):

    #remove nans
    flux = og_flux[~np.isnan(og_flux)]
    wavelength = og_wavelength[~np.isnan(og_flux)]

    min_flux, max_flux = min(flux), max(flux)

    flux_levels = np.linspace(min_level * (min_flux),
                              max_level * (max_flux),
                              num_levels)

    bisector_wavelengths = []
    bisector_fluxes = []
    widths = []

    for level in flux_levels:
        # Find all crossing points
        cross_points = np.where(np.diff(np.sign(flux - level)))[0]

        if len(cross_points) >= 2:
            left_idx, right_idx = cross_points[0], cross_points[-1]
            # Interpolate to find precise wavelength values
            left_wave = interpolate_crossing(wavelength[left_idx:left_idx+4], 
                                             flux[left_idx:left_idx+4], level)
            right_wave = interpolate_crossing(wavelength[right_idx:right_idx+4], 
                                              flux[right_idx:right_idx+4], level)

            # Calculate bisector point
            bisector_wave = (left_wave + right_wave) / 2
            bisector_wavelengths.append(bisector_wave)
            bisector_fluxes.append(level)
            widths.append(right_wave - left_wave)


    return np.array(bisector_wavelengths), np.array(bisector_fluxes)