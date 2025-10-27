import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial

def extend_lines(Is, wl, rest_wl, n_lines, shift=0.3, jitter_scale=0.05, template_width=1e-4):
    """
    Extend each time-step spectrum by duplicating the line `n_lines` times with random wavelength shifts,
    and generate a continuous (interpolated) template mask.

    Parameters
    ----------
    Is : np.ndarray
        2D array of shape (n_times, n_wl), normalized intensity spectra.
    wl : np.ndarray
        1D array of shape (n_wl,), wavelength axis for one line.
    rest_wl : float
        Central (rest) wavelength of the line.
    n_lines : int
        Number of copies (including the original) per time step.
    shift : float
        Separation between successive lines (in same units as `wl`).
    jitter_scale : float
        Standard deviation of small random wavelength jitter for each line.
    template_width : float
        Width (sigma) of each line in the template (in wavelength units).

    Returns
    -------
    ext_wl : np.ndarray
        Extended wavelength array containing all shifted lines.
    ext_I : np.ndarray
        Extended intensity array of shape (n_times, len(ext_wl)).
    template : np.ndarray
        Interpolated continuous template mask over `ext_wl`.
    """

    n_times, n_wl = np.array(Is).shape
    wl_extended = []
    Is_extended = np.zeros((n_times, n_lines * n_wl))

    template_centers = np.zeros(n_lines)

    for j in range(n_lines):
        # Random jitter avoids aliasing in the combined spectrum
        jitter = np.random.normal(0, jitter_scale)
        wl_shifted = wl + j * shift + jitter
        wl_extended.append(wl_shifted)
        Is_extended[:, j*n_wl:(j+1)*n_wl] = Is
        template_centers[j] = rest_wl + j * shift + jitter

    wl_extended = np.concatenate(wl_extended)

    # Create uniform wavelength grid
    step = np.median(np.diff(wl))
    ext_wl = np.arange(wl_extended.min(), wl_extended.max(), step)

    # Interpolate intensity spectra onto uniform grid
    ext_I = np.zeros((n_times, len(ext_wl)))
    for i in range(n_times):
        f_I = interp1d(wl_extended, Is_extended[i], kind='linear', bounds_error=False, fill_value=1)
        ext_I[i] = f_I(ext_wl)

    # ---- Interpolated template generation ----
    # Use Gaussian kernels for each line center, then sum and normalize
    template = np.zeros_like(ext_wl)
    for center in template_centers:
        template += np.exp(-0.5 * ((ext_wl - center) / template_width)**2)

    # Normalize template so its peak = 1
    template /= template.max()

    return ext_wl, ext_I, template

def get_ccf(wl, flux, template_flux):
    '''
    Compute the cross-correlation function (CCF) between a given flux spectrum and a template spectrum.
    '''

    even_wl = np.linspace(wl[0], wl[-1], len(wl))
    flux = np.interp(even_wl, wl, flux) #interpolate both to even wavelength grid
    template_flux = np.interp(even_wl, wl, template_flux)

    c = 299792458.0 #speed of light in m/s

    flux_fft = np.fft.fft(flux)
    template_flux_fft = np.fft.fft(template_flux)

    corr = np.fft.ifft(flux_fft * np.conj(template_flux_fft)) #cross correlating 
    corr_shifted = np.fft.fftshift(corr) #shift the zero-frequency component to the center of the spectrum.
    #the peak of this should occur where the two profiles are most similar - giving shift between them

    grid_midpoint = even_wl[len(wl)//2] #this is the centre of wavelength grid 

    vel = (even_wl - grid_midpoint) * c / grid_midpoint #this shifts the grid to be centered at 0 and converts to velocity

    return np.real(vel), np.real(corr_shifted)

def compute_equivalent_width(wl, I, continuum=None):
    """Compute equivalent width by integrating 1 - I/continuum over wavelength."""
    if continuum is None:
        continuum = np.max(I)
    y = 1.0 - (I / continuum)
    return np.trapz(y, wl)

def find_line_core(x, y, window=2, deg=2):
    """
    Find the line core (x, y) using a local polynomial fit around the minimum.
    
    Parameters
    ----------
    x : 1D ndarray
        X-axis values (e.g., wavelength or velocity).
    y : 1D ndarray
        Flux values (normalized line profile).
    window : int
        Number of points on each side of the minimum to include in the fit.
    deg : int
        Degree of polynomial to fit (2 for quadratic minimum).
        
    Returns
    -------
    x_core : float
        X position of the line minimum (line core).
    y_core : float
        Y value at the line minimum.
    """
    # --- find approximate minimum index ---
    min_idx = np.argmin(y)
    
    # --- select points around minimum ---
    start = max(min_idx - window, 0)
    end   = min(min_idx + window + 1, len(y))
    x_fit = x[start:end]
    y_fit = y[start:end]

    
    # --- fit polynomial ---
    p = Polynomial.fit(x_fit, y_fit, deg)

    # --- find minimum of polynomial ---
    # derivative = 0 at minimum
    dp = p.deriv()
    roots = dp.roots()
    
    # select root within fitting window
    real_roots = roots[np.isreal(roots)].real
    if len(real_roots) == 0:
        # fallback: take minimal value in fitted points
        idx_min = np.argmin(y_fit)
        x_core = x_fit[idx_min]
        y_core = y_fit[idx_min]
    else:
        # pick root closest to original minimum
        x_core = real_roots[np.argmin(np.abs(real_roots - x[min_idx]))]
        y_core = p(x_core)
    
    return x_core, y_core


def compute_core_depth(wl, I, norm = True):
    """Core depth = continuum - line core."""
    if norm == True:
        I = I / np.max(I)
        
    core = find_line_core(wl, I)[1]
    return np.max(I) - core

def compute_fwhm(wl, I, num = 2.0):
    """Compute the full width at half maximum (FWHM) of a line profile."""
    continuum = np.max(I)
    core = np.min(I)
    half_level = core + (continuum - core) / num
    core_idx = np.argmin(I)

    # left crossing
    left_idx = np.max(np.where(I[:core_idx] > half_level)[0])
    x1 = np.interp(half_level, I[left_idx:left_idx+2][::-1], wl[left_idx:left_idx+2][::-1])

    # right crossing
    right_idx = core_idx + np.min(np.where(I[core_idx:] > half_level)[0])
    x2 = np.interp(half_level, I[right_idx-1:right_idx+1], wl[right_idx-1:right_idx+1])

    fwhm_lambda = np.abs(x2 - x1)
    return fwhm_lambda

def get_rv(wl, flux, template_flux):
    '''
    gets relative radial velocity (m/s) of the given flux profile to a template profile, using fft ccf
    # '''

    #ensure that wl grid is evenly spaced
    even_wl = np.linspace(wl[0], wl[-1], len(wl))
    even_flux = np.interp(even_wl, wl, flux)
    even_template_flux = np.interp(even_wl, wl, template_flux)

    # Extend wl, flux, and template_flux arrays - ensuring continuum never goes off the grid 
    extended_wl = np.concatenate((even_wl[0] - np.arange(50)[::-1], even_wl, even_wl[-1] + np.arange(1, 51)))
    extended_flux = np.concatenate((np.full(50, even_flux[0]), even_flux, np.full(50, even_flux[-1])))
    extended_template_flux = np.concatenate((np.full(50, even_template_flux[0]), even_template_flux, np.full(50, even_template_flux[-1])))

    c = 299792458.0

    flux_fft = np.fft.fft(extended_flux)
    template_flux_fft = np.fft.fft(extended_template_flux)

    corr = np.fft.ifft(flux_fft * np.conj(template_flux_fft)) #cross correlating 
    corr_shifted = np.fft.fftshift(corr) #shift the zero-frequency component to the center of the spectrum.
    #the peak of this should occur where the two profiles are most similar - giving shift between them

    grid_midpoint = extended_wl[len(extended_wl)//2] #this is the centre of wavelength grid 

    vel = (extended_wl - grid_midpoint) * c / grid_midpoint #this shifts the grid to be centered at 0 and converts to velocity

    # Isolate the section of the ccf that contains the peak
    peak_index = np.argmax(corr_shifted)
    corr_shifted = corr_shifted[peak_index-1:peak_index+2]

    # Normalize corr_shifted to go from 0 to 1
    corr_shifted /= np.max(corr_shifted)
    
    p = Polynomial.fit(vel[peak_index-1:peak_index+2], corr_shifted, 2)
    c, b, a = p.convert()
    x_maxima = -b / (2 * a)
    rv = x_maxima

    return rv  

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

def calculate_line_bisector(og_wavelength, og_flux, num_levels=50, lower = 0.1, upper = 0.05, return_width = False):

    #remove nans
    flux = og_flux[~np.isnan(og_flux)]
    wavelength = og_wavelength[~np.isnan(og_flux)]
    
    # Define flux levels based on the actual flux range (cutting the bottom and top 10%)
    min_flux, max_flux = min(flux), max(flux)
    # flux_levels = np.linspace(min_flux + 0.15 * (max_flux - min_flux),
    #                           min_flux + 0.85 * (max_flux - min_flux),
    #                           num_levels)

    depth = max_flux - min_flux
    lower_bound = min_flux + lower * depth
    upper_bound = max_flux - upper * depth

    flux_levels = np.linspace(lower_bound,
                              upper_bound,
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


    if return_width:
        return np.array(bisector_wavelengths), np.array(bisector_fluxes), np.array(widths)
    
    else:
        return np.array(bisector_wavelengths), np.array(bisector_fluxes)







