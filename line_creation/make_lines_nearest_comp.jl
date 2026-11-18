using CUDA
using PyPlot
using Printf

include(joinpath(@__DIR__, "make_lines.jl"))
module NearestDISCO
include(joinpath(@__DIR__, "make_lines_nearest.jl"))
end

const PATCH_ID = Int32(1)
const EPOCH_SEED = UInt32(0)
const PROFILE_MODES = (
    (Int32(1), "original"),
    (Int32(2), "centered"),
    (Int32(3), "fixed_cb"),
)

@inline function profile_value(
    p::DISCOJulia.DISCOParamsDevice, mu, wave, patch_id, epoch_seed, mode
)
    if mode == Int32(1)
        return DISCOJulia.disco_intensity_original(
            p, mu, wave, patch_id, epoch_seed
        )
    elseif mode == Int32(2)
        return DISCOJulia.disco_intensity_centered(
            p, mu, wave, patch_id, epoch_seed
        )
    end
    return DISCOJulia.disco_intensity_fixed_cb(
        p, mu, wave, patch_id, epoch_seed
    )
end

@inline function profile_value(
    p::NearestDISCO.DISCOJulia.DISCOParamsDevice, mu, wave, patch_id, epoch_seed, mode
)
    if mode == Int32(1)
        return NearestDISCO.DISCOJulia.disco_intensity_original(
            p, mu, wave, patch_id, epoch_seed
        )
    elseif mode == Int32(2)
        return NearestDISCO.DISCOJulia.disco_intensity_centered(
            p, mu, wave, patch_id, epoch_seed
        )
    end
    return NearestDISCO.DISCOJulia.disco_intensity_fixed_cb(
        p, mu, wave, patch_id, epoch_seed
    )
end

function profile_kernel!(out, p, mu, patch_id, epoch_seed, mode)
    wave = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if wave <= length(out)
        @inbounds out[wave] = profile_value(
            p, mu, Int32(wave), patch_id, epoch_seed, mode
        )
    end
    return nothing
end

function get_profile(p, mu::Float32, mode::Int32)
    out = CUDA.zeros(Float32, Int(p.n_wave))
    threads = 256
    blocks = cld(length(out), threads)
    CUDA.@sync @cuda threads=threads blocks=blocks profile_kernel!(
        out, p, mu, PATCH_ID, EPOCH_SEED, mode
    )
    return Array(out)
end

function normalize_profile(profile, mu, mode_name)
    scale = maximum(profile)
    isfinite(scale) && scale > 0 ||
        error("Invalid continuum scale for mode=$(mode_name), μ=$(mu): $(scale)")
    return profile ./ scale
end

function nearest_sample(grid::Vector{Float32}, mu::Float32)
    return grid[argmin(abs.(grid .- mu))]
end

function compare_profiles(
    original, nearest, nearest_grid, wavelengths, mus, output_prefix
)
    modes = Dict{String,Vector{Tuple{Float32,Vector{Float32},Vector{Float32}}}}()
    summary_path = output_prefix * "_summary.csv"
    mkpath(dirname(abspath(summary_path)))

    open(summary_path, "w") do io
        println(io, "mode,mu,nearest_mu,rms_normalized,max_abs_normalized")
        for (mode, mode_name) in PROFILE_MODES
            mode_profiles = Tuple{Float32,Vector{Float32},Vector{Float32}}[]
            println("\nProfile mode: $(mode_name)")
            for mu in mus
                original_profile = normalize_profile(
                    get_profile(original, mu, mode), mu, mode_name
                )
                nearest_profile = normalize_profile(
                    get_profile(nearest, mu, mode), mu, mode_name
                )
                residual = nearest_profile .- original_profile
                rms = sqrt(sum(abs2, residual) / length(residual))
                max_abs = maximum(abs.(residual))
                sampled_mu = nearest_sample(nearest_grid, mu)
                push!(mode_profiles, (mu, original_profile, nearest_profile))
                @printf(
                    "μ=%0.6f -> nearest μ=%0.6f | RMS=%0.4e | max=%0.4e\n",
                    mu, sampled_mu, rms, max_abs
                )
                @printf(
                    io, "%s,%.8f,%.8f,%.9e,%.9e\n",
                    mode_name, mu, sampled_mu, rms, max_abs
                )
            end
            modes[mode_name] = mode_profiles
        end
    end

    # Plot all selected knots and interval midpoints to show where the lookup
    # agrees at sampled μ and how it differs between samples.
    for (mode, mode_name) in PROFILE_MODES
        profiles = modes[mode_name]
        fig, (ax_profile, ax_residual) = plt.subplots(
            2, 1; figsize=(11, 8), sharex=true,
            gridspec_kw=Dict("height_ratios" => [3, 1]),
        )
        for (mu, original_profile, nearest_profile) in profiles
            residual = nearest_profile .- original_profile
            ax_profile.plot(
                wavelengths, original_profile; alpha=0.65, linewidth=1.0,
                label="dense μ=$(round(mu; digits=3))",
            )
            ax_profile.plot(
                wavelengths, nearest_profile; alpha=0.8, linewidth=1.0,
                linestyle="--", label="nearest μ=$(round(mu; digits=3))",
            )
            ax_residual.plot(wavelengths, residual; linewidth=0.9, label="μ=$(round(mu; digits=3))")
        end
        ax_profile.set_ylabel("Normalized intensity")
        ax_profile.set_title("Dense vs nearest DISCO: $(mode_name)")
        ax_profile.grid(true; alpha=0.3)
        ax_profile.legend(loc="best", fontsize=7, ncol=2)
        ax_residual.set_xlabel("Wavelength (Å)")
        ax_residual.set_ylabel("Nearest - dense")
        ax_residual.axhline(0.0; color="black", linestyle=":", alpha=0.7)
        ax_residual.grid(true; alpha=0.3)
        fig.tight_layout()
        figure_path = output_prefix * "_$(mode_name).png"
        fig.savefig(figure_path; dpi=200)
        close(fig)
        println("Plot saved: $(figure_path)")
    end
    println("Summary saved: $(summary_path)")
end

function main(args)
    length(args) >= 2 || error(
        "Usage: julia make_lines_nearest_comp.jl ORIGINAL.h5 NEAREST.h5 " *
        "[OUTPUT_PREFIX] [additional_mu ...]"
    )
    original_path, nearest_path = args[1], args[2]
    default_output_prefix = joinpath(@__DIR__, "dense_vs_nearest")
    parsed_third_arg = length(args) >= 3 ? tryparse(Float32, args[3]) : nothing
    has_output_prefix = length(args) >= 3 && parsed_third_arg === nothing
    output_prefix = has_output_prefix ? args[3] : default_output_prefix
    mu_arg_start = has_output_prefix ? 4 : 3
    additional_mus = length(args) >= mu_arg_start ?
        parse.(Float32, args[mu_arg_start:end]) : Float32[]

    CUDA.functional() || error("A functional CUDA device is required")
    original_host = DISCOJulia.DISCOParams(original_path)
    nearest_host = NearestDISCO.DISCOJulia.DISCOParams(nearest_path)
    original_host.wavelength == nearest_host.wavelength ||
        error("The two HDF5 files have different wavelength grids")

    original = DISCOJulia.to_device(original_host)
    nearest = NearestDISCO.DISCOJulia.to_device(nearest_host)
    grid = nearest_host.mu_grid
    points = Float32[(grid[i] + grid[i + 1]) / 2 for i in 1:(length(grid) - 1)]
    mus = sort!(unique!(vcat(copy(grid), points, additional_mus)))
    all(mu -> grid[1] <= mu <= grid[end], mus) ||
        error("All comparison μ values must lie inside the nearest HDF5 μ range")

    println("Original HDF5: $(original_path)")
    println("Nearest HDF5:  $(nearest_path)")
    println("Patch ID: $(PATCH_ID); epoch seed: $(EPOCH_SEED) (deterministic median mode)")
    println("Nearest μ grid: $(grid)")
    println("Comparing $(length(mus)) μ values across $(length(PROFILE_MODES)) profile modes")

    compare_profiles(
        original, nearest, grid, original_host.wavelength .* 10f0, mus, output_prefix
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
