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
        prof_d[w] = DISCOJulia.disco_intensity(p_device, mu, Int32(w), patch_id, epoch_seed)
    end
    return nothing
end

function get_disco_profile_gpu(p_device::DISCOParamsDevice, mu::Float32; 
                                patch_id::Int32=Int32(1), epoch_seed::UInt32=UInt32(0))
    n_wave = Int(p_device.n_wave)
    prof_d = CUDA.zeros(Float32, n_wave)
    
    threads = 256
    blocks = cld(n_wave, threads)
    
    @cuda threads=threads blocks=blocks kernel_single_profile!(prof_d, p_device, mu, patch_id, epoch_seed)
    CUDA.synchronize()
    
    return Array(prof_d) 
end

function extract_cb(h5_path::String)
    p_host = DISCOParams(h5_path)
    p_device = to_device(p_host)

    MAX_DEG = 83.0
    MU_MIN = Float64(cos(deg2rad(MAX_DEG)))
    MU_MAX = 1.00
    N_MU_DENSE = 4096
    mus = range(MU_MIN, MU_MAX, N_MU_DENSE)

    wl_nm = p_host.wavelength
    wl_angstroms = wl_nm .* 10.0f0  

    jl_cb = Vector{Float64}(undef,size(mus)...)

    for mu in 1:length(mus)
        jl_prof = get_disco_profile_gpu(p_device, Float32(mus[mu]); epoch_seed=UInt32(0))
        jl_prof = jl_prof ./ maximum(jl_prof)

        jl_cs = CubicSpline(wl_angstroms, jl_prof)
        wl_new = range(ref_wl - 0.1, ref_wl + 0.1, 500)
        jl_prof_new = jl_cs(wl_new)

        jl_cb[mu] = wl_new[argmin(jl_prof_new)] - ref_wl
    end

    df = DataFrame(Column1 = (jl_cb ./ ref_wl) .* (3*10^8))

    CSV.write("$(line).csv", df)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    H5_PATH = "/storage/home/efg5335/work/sw/DISCO/GRASS/data/disco_$(line)_params.h5"
    extract_cb(H5_PATH)
end

