import numpy as np
from scipy.ndimage import gaussian_filter1d

def _compute_edges(centers):
    """
    Compute pixel edges from pixel centers by midpoint rule.
    Returns array of length len(centers)+1.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        raise ValueError("Need at least two wavelength points to compute edges.")
    diffs = np.diff(centers)
    # interior edges
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    # extend first and last by nearest spacing
    edges[0] = centers[0] - 0.5 * diffs[0]
    edges[-1] = centers[-1] + 0.5 * diffs[-1]
    return edges

def _flux_conserving_rebin(in_centers, in_flux_density, out_centers,
                           in_edges=None, out_edges=None):
    """
    Flux-conserving rebin from arbitrary input centers -> arbitrary output centers.

    in_centers, out_centers : 1D arrays (wavelengths)
    in_flux_density : flux per unit wavelength at input centers (same units)
    returns: out_flux_density at out_centers (flux per unit wavelength)
    """
    in_centers = np.asarray(in_centers, dtype=float)
    out_centers = np.asarray(out_centers, dtype=float)
    in_flux_density = np.asarray(in_flux_density, dtype=float)

    if in_edges is None:
        in_edges = _compute_edges(in_centers)
    else:
        in_edges = np.asarray(in_edges, dtype=float)
    if out_edges is None:
        out_edges = _compute_edges(out_centers)
    else:
        out_edges = np.asarray(out_edges, dtype=float)

    nin = in_centers.size
    nout = out_centers.size
    in_widths = (in_edges[1:] - in_edges[:-1])
    out_widths = (out_edges[1:] - out_edges[:-1])

    # total flux in each input pixel
    in_flux_total = in_flux_density * in_widths

    out_flux_total = np.zeros(nout, dtype=float)

    i = 0  # pointer into input pixels
    for j in range(nout):
        out_l = out_edges[j]
        out_r = out_edges[j+1]

        # advance i until input pixel might overlap output pixel
        while (i < nin - 1) and (in_edges[i+1] <= out_l):
            i += 1

        k = i
        # accumulate contributions from overlapping input pixels
        while (k < nin) and (in_edges[k] < out_r):
            in_l = in_edges[k]
            in_r = in_edges[k+1]
            overlap_l = max(in_l, out_l)
            overlap_r = min(in_r, out_r)
            if overlap_r > overlap_l:
                overlap = overlap_r - overlap_l
                # portion of input pixel's flux that falls into this output bin
                if in_widths[k] > 0:
                    frac = overlap / in_widths[k]
                    out_flux_total[j] += in_flux_total[k] * frac
            k += 1

    # convert total flux in each output bin back to flux density
    # (avoid dividing by zero width)
    out_flux_density = np.zeros_like(out_flux_total)
    nonzero = out_widths > 0
    out_flux_density[nonzero] = out_flux_total[nonzero] / out_widths[nonzero]
    return out_flux_density

def degrade_and_rebin(wl_nm, flux, R=190000, pixels_per_fwhm=4,
                              conv_oversample=10, kernel_nsigma=6):
    """
    Degrade a normalized stellar absorption line to a Gaussian LSF of resolving
    power R (default 190,000) and flux-conserving
    rebin onto a common log-wavelength grid with ~pixels_per_fwhm
    sampling of the LSF FWHM.

    Parameters
    ----------
    wl_nm : 1D array
        Wavelengths in nanometers (must be strictly increasing).
    flux : 1D array
        Normalized flux (flux density per nm) at wl_nm.
    R : float
        Resolving power R = lambda / Delta_lambda (default 190000).
    pixels_per_fwhm : int
        Target number of output pixels per LSF FWHM (default 4).
    conv_oversample : int
        Internal oversampling factor when building the convolution grid:
        the convolution grid step in ln(lambda) will be (1/R)/conv_oversample.
        Higher -> more accurate convolution (default 10).
    kernel_nsigma : float
        Number of sigma to include as padding for the convolution grid
        (default 6).

    Returns
    -------
    wl_common : 1D array
        Common output wavelength grid (nm).
    flux_orig_common : 1D array
        The original input spectrum rebinned (flux density per nm) onto wl_common
        using a flux-conserving algorithm (Carnall 2017 style).
    flux_degraded_common : 1D array
        The degraded (LSF-convolved) spectrum rebinned onto wl_common.
    """
    # -------------------
    # 1) basic checks & sorting
    wl_nm = np.asarray(wl_nm, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if wl_nm.ndim != 1 or flux.ndim != 1 or wl_nm.size != flux.size:
        raise ValueError("wl_nm and flux must be 1D arrays of the same length.")
    # sort
    order = np.argsort(wl_nm)
    wl_nm = wl_nm[order]
    flux = flux[order]

    # compute input edges for flux-conserving rebinning later
    in_edges = _compute_edges(wl_nm)

    # -------------------
    # 2) build convolution grid in ln(lambda)

    x = np.log(wl_nm)                # natural log of wavelength
    x_min, x_max = x[0], x[-1]

    sigma_x = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)) * R)

    # dx for the convolution grid (log-lambda) chosen so that there are
    # conv_oversample samples per FWHM (FWHM_x = 1/R)
    dx_conv = 1.0 / (R * float(conv_oversample))

    # extend the grid by kernel_nsigma * sigma_x on both sides to avoid
    # edge truncation
    x_conv = np.arange(x_min - kernel_nsigma * sigma_x,
                       x_max + kernel_nsigma * sigma_x + dx_conv,
                       dx_conv)
    lam_conv = np.exp(x_conv)  # wavelengths (nm) at convolution grid centers

    # -------------------
    # 3) resample original spectrum onto convolution lambda grid (flux conserving)
    #    The input flux is per nm (flux density); _flux_conserving_rebin handles that.
    flux_on_conv = _flux_conserving_rebin(wl_nm, flux, lam_conv,
                                          in_edges=in_edges,
                                          out_edges=_compute_edges(lam_conv))

    # convert flux density per nm -> flux per unit ln(lambda) for convolution
    # because dλ = λ d(ln λ)  =>  F_ln = F_lambda * λ
    flux_per_x = flux_on_conv * lam_conv

    # -------------------
    # 4) build Gaussian kernel in x (log-lambda) domain and convolve
    sigma_pix = sigma_x / dx_conv
    half_width_pix = int(np.ceil(8.0 * sigma_pix))  # ±8 sigma
    pix_idx = np.arange(-half_width_pix, half_width_pix + 1, dtype=float)
    pos = pix_idx * dx_conv
    kernel = np.exp(-0.5 * (pos / sigma_x) ** 2)
    kernel /= kernel.sum()  # normalize discrete kernel to 1

    # discrete convolution (same-length, kernel normalized)
    conv_flux_per_x = np.convolve(flux_per_x, kernel, mode='same')

    # convert back to flux density per nm:
    flux_conv_on_grid = conv_flux_per_x / lam_conv

    # -------------------
    # 5) define common output grid with ~pixels_per_fwhm per FWHM
    # choose dx_common in ln(lambda) so that FWHM_x = 1/R has pixels_per_fwhm samples:
    dx_common = 1.0 / (R * float(pixels_per_fwhm))
    # choose common grid spanning input wavelengths
    x_common = np.arange(x_min, x_max + 0.5 * dx_common, dx_common)
    wl_common = np.exp(x_common)

    # -------------------
    # 6) flux-conserving rebin both original and degraded onto wl_common
    # compute edges for conv grid and for common grid
    conv_edges = _compute_edges(lam_conv)
    common_edges = _compute_edges(wl_common)

    flux_orig_common = _flux_conserving_rebin(wl_nm, flux, wl_common,
                                              in_edges=in_edges,
                                              out_edges=common_edges)
    flux_degraded_common = _flux_conserving_rebin(lam_conv, flux_conv_on_grid,
                                                  wl_common,
                                                  in_edges=conv_edges,
                                                  out_edges=common_edges)

    return wl_common[10:-10], flux_degraded_common[10:-10]


def add_photon_noise(I, snr):
    """
    Add photon noise to normalized spectra.

    Parameters
    ----------
    I : 2D ndarray
        Input spectra (normalised).
    snr : float
        Desired signal-to-noise ratio per pixel (at continuum).

    Returns
    -------
    noisy_profiles : 2D ndarray
        Spectra with Gaussian photon noise added.
    """
    sigma = I / snr
    noise = np.random.normal(0, sigma, I.shape)
    return I + noise
