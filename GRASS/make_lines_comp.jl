# make_lines_example.jl
# Comparison and Residual Plotter: DISCO Python vs. DISCO Julia GPU Module

using PyCall
using PyPlot
using HDF5
using CUDA
using Printf

# Include DISCO Julia module
include(joinpath(@__DIR__, "src", "DISCOJulia.jl"))
using .DISCOJulia

line = "Fe5250"

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
    
    return Array(prof_d) # Single clean DtoH copy
end

"""
    compare_disco_python_julia(h5_path::String, disco_root::String)

Validates point-by-point agreement between DISCO Python (LineProfileGenerator)
and DISCO Julia (DISCOJulia) across multiple viewing angles (μ = 1.0, 0.8, 0.5, 0.2).
Plots profiles, residuals (Julia - Python), and prints RMS errors.
"""
function compare_disco_python_julia(h5_path::String, disco_root::String)
    # 1. Initialize Python DISCO via PyCall
    pyimport("sys")["path"].insert(0, joinpath(disco_root, "line_creation"))
    make_lines = pyimport("make_lines")
    py_gen = make_lines.LineProfileGenerator(line)

    # 2. Initialize Julia DISCO parameters
    p_host = DISCOParams(h5_path)
    p_device = to_device(p_host)

    mus = Float32[1.0, 0.8, 0.5, 0.2]
    wl_nm = p_host.wavelength
    wl_angstroms = wl_nm .* 10.0f0  # Convert nm to Å for display

    # Setup 2-panel figure: Top = Profiles, Bottom = Residuals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=true, gridspec_kw=Dict("height_ratios" => [3, 1]))

    println("=================================================================")
    println(" DISCO Python vs. Julia Profile Comparison & Residual Summary")
    println("=================================================================")

    for mu in mus
        deg = Float64(rad2deg(acos(mu)))

        # A. Python DISCO reference (median profiles, quantiles = 0.5)
        py_res = py_gen.get_full_profile(deg, GT_quantile=0.5, ratio_quantile=0.5)
        py_prof = Float32.(py_res[1])
        py_prof = py_prof ./ maximum(py_prof)

        # B. Julia DISCO median evaluation (epoch_seed = 0 triggers exact median mode)
        jl_prof = get_disco_profile_gpu(p_device, mu; epoch_seed=UInt32(0))
        jl_prof = jl_prof ./ maximum(jl_prof)

        # C. Compute Residuals
        residual = jl_prof .- py_prof
        rms_err = sqrt(sum(residual .^ 2) / length(residual))
        max_err = maximum(abs.(residual))

        @printf("μ = %3.1f (deg = %5.1f°) | RMS Residual: %.2e | Max Residual: %.2e\n", 
                mu, deg, rms_err, max_err)

        # D. Plotting
        ax1.plot(wl_angstroms, py_prof, label="μ = $mu (Python)", linestyle="-", alpha=0.8)
        ax1.plot(wl_angstroms, jl_prof, label="μ = $mu (Julia)", linestyle="--", alpha=0.8)
        ax2.plot(wl_angstroms, residual, label="μ = $mu Residual")
    end

    ax1.set_ylabel("Normalized Intensity")
    ax1.set_title("$line Å Center-to-Limb Line Profiles: Python vs. Julia")
    ax1.legend(loc="lower right", fontsize=8, ncol=2)
    ax1.grid(true, alpha=0.3)

    ax2.set_xlabel("Wavelength (Å)")
    ax2.set_ylabel("Residual (Julia - Python)")
    ax2.axhline(0.0, color="black", linestyle=":", alpha=0.7)
    ax2.grid(true, alpha=0.3)
    ax2.set_ylim(-1e-4, 1e-4)

    plt.tight_layout()
    out_fig = joinpath(@__DIR__, "disco_python_vs_julia_residuals_$(line).png")
    plt.savefig(out_fig, dpi=300)
    println("=================================================================")
    println(" Residual plot saved to: ", out_fig)
    println("=================================================================")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    H5_PATH = "/storage/home/efg5335/work/sw/DISCO/GRASS/data/disco_$(line)_params.h5"
    DISCO_ROOT = "/storage/home/efg5335/work/sw/DISCO"
    compare_disco_python_julia(H5_PATH, DISCO_ROOT)
end

