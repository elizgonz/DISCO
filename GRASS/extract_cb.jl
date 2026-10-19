using PyCall
using CSV, DataFrames
using PyPlot
using HDF5
using CUDA
using Printf
using PyCall

py"""
import sys
sys.path.append('.')
"""
scipy_interp = pyimport("scipy.interpolate")
CubicSpline = scipy_interp.CubicSpline

include(joinpath(@__DIR__, "src", "DISCOJulia.jl"))
using .DISCOJulia

line = "Fe6173"
ref_wl = 6173.3344

function kernel_single_profile!(prof_d, p_device, mu, patch_id, epoch_seed)
    w = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if w <= length(prof_d)
        prof_d[w] = DISCOJulia.disco_intensity(
            p_device, mu, Int32(w), patch_id, epoch_seed
        )
    end
    return nothing
end

function get_disco_profile_gpu(
    p_device::DISCOJulia.DISCOParamsDevice,
    mu::Float32;
    patch_id::Int32=Int32(1),
    epoch_seed::UInt32=UInt32(0),
)
    n_wave = Int(p_device.n_wave)
    prof_d = CUDA.zeros(Float32, n_wave)
    
    threads = 256
    blocks = cld(n_wave, threads)
    
    @cuda threads=threads blocks=blocks kernel_single_profile!(prof_d, p_device, mu, patch_id, epoch_seed)
    CUDA.synchronize()
    
    return Array(prof_d) 
end

function refined_minimum_offset(profile, wavelengths, ref_wl)
    core_idx = argmin(profile)
    correction = 0.0

    if 1 < core_idx < length(profile)
        left = Float64(profile[core_idx - 1])
        center = Float64(profile[core_idx])
        right = Float64(profile[core_idx + 1])
        curvature = left - 2.0 * center + right
        if curvature > 0.0
            correction = clamp(0.5 * (left - right) / curvature, -1.0, 1.0)
        end
    end

    wavelength_step = step(wavelengths)
    return wavelengths[core_idx] + correction * wavelength_step - ref_wl
end

function extract_cb(h5_path::String)
    p_host = DISCOJulia.DISCOParams(h5_path)
    p_device = DISCOJulia.to_device(p_host)

    MU_MIN = 0.0
    MU_MAX = 1.00
    N_MU_DENSE = 4096
    mus = range(MU_MIN, MU_MAX, N_MU_DENSE)

    wl_nm = p_host.wavelength
    wl_angstroms = wl_nm .* 10.0f0  

    jl_cb = Vector{Float64}(undef, length(mus))
    polynomial_cb = Vector{Float64}(undef, length(mus))
    original_cb = Union{Missing,Float64}[missing for _ in eachindex(mus)]

    reference_nm = ref_wl / 10
    for i in eachindex(mus)
        mu = Float32(mus[i])
        jl_prof = get_disco_profile_gpu(p_device, mu; epoch_seed=UInt32(0))
        jl_prof = jl_prof ./ maximum(jl_prof)

        jl_cs = CubicSpline(wl_angstroms, jl_prof)
        wl_new = range(ref_wl - 0.1, ref_wl + 0.1; length=500)
        jl_prof_new = jl_cs(wl_new)

        jl_cb[i] = refined_minimum_offset(jl_prof_new, wl_new, ref_wl)
        polynomial_cb[i] =
            DISCOJulia.cb_polynomial(p_host, mu) / reference_nm * 3e8
        if mu >= p_host.mu_min
            original_cb[i] =
                DISCOJulia.original_cb_offset(p_host, mu) / reference_nm * 3e8
        end
    end

    df = DataFrame(
        Mu = Float64.(mus),
        DISCOCB = original_cb,
        PolynomialCB = polynomial_cb,
        MeasuredCB = (jl_cb ./ ref_wl) .* 3e8,
        Poly_c0_nm = fill(p_host.cb_poly_c0_nm, length(mus)),
        Poly_c1_nm_per_mu = fill(p_host.cb_poly_c1_nm_per_mu, length(mus)),
        Poly_c2_nm_per_mu2 = fill(p_host.cb_poly_c2_nm_per_mu2, length(mus)),
    )

    CSV.write("data/disco_$(line)_cb.csv", df)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    H5_PATH = "/storage/home/efg5335/work/sw/DISCO/GRASS/data/disco_$(line)_params.h5"
    extract_cb(H5_PATH)
end
