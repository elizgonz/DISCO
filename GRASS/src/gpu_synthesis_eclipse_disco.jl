using CUDA

# This file and disco_fe5250.jl should both be in GRASS/src/.
include(joinpath(@__DIR__, "disco_fe5250.jl"))
using .DISCOFe5250

function kernel_synthesize_eclipse_disco!(
    disk_flux, mu_grid, rv_grid, weight_grid,
    n_patches::Int32, p::DISCOParamsDevice,
    wav_min::Float32, wav_max::Float32,
    epoch_seed::UInt32,
)
    patch = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    patch > Int(n_patches) && return

    @inbounds begin
        w = weight_grid[patch]
        w <= 0f0 && return

        mu = mu_grid[patch]
        rv = rv_grid[patch]

        beta = rv / 299792.458f0
        shift = sqrt((1f0 + beta) / (1f0 - beta))

        n_obs = length(disk_flux)
        rest_lo = p.wavelength[1]
        rest_hi = p.wavelength[Int(p.n_wave)]
        rest_step = (rest_hi - rest_lo) / Float32(p.n_wave - Int32(1))
        obs_step = (wav_max - wav_min) / Float32(n_obs - 1)

        j = 1
        while j <= n_obs
            λobs = wav_min + Float32(j - 1) * obs_step
            λrest = λobs / shift
            pos = (λrest - rest_lo) / rest_step

            if pos >= 0f0 && pos <= Float32(p.n_wave - Int32(1))
                k = Int32(floor(pos))
                k = min(k, p.n_wave - Int32(2))
                t = pos - Float32(k)

                f0 = disco_intensity(
                    p, mu, k + Int32(1), Int32(patch), epoch_seed
                )
                f1 = disco_intensity(
                    p, mu, k + Int32(2), Int32(patch), epoch_seed
                )

                value = f0 + t * (f1 - f0)
                contribution = w * value
                CUDA.@atomic disk_flux[j] += contribution
            end

            j += 1
        end
    end

    return
end

"""Synthesize a DISCO eclipse spectrum from GPU-resident patch arrays."""
function synthesize_eclipse_disco(mu_d::CuArray{Float32,1},
                                  rv_d::CuArray{Float32,1},
                                  weight_d::CuArray{Float32,1},
                                  p::DISCOParamsDevice;
                                  wav_min::Float32,
                                  wav_max::Float32,
                                  n_wave::Int=length(p.wavelength),
                                  epoch_seed::UInt32=UInt32(0x12345678),
                                  threads_per_block::Int=128)
    n = length(mu_d)
    @assert length(rv_d) == n && length(weight_d) == n
    @assert n_wave >= 2
    flux = CUDA.zeros(Float32, n_wave)
    threads = threads_per_block
    blocks = cld(n, threads)

    @cuda threads=threads blocks=blocks kernel_synthesize_eclipse_disco!(
        flux,
        mu_d,
        rv_d,
        weight_d,
        Int32(n),
        p,
        wav_min,
        wav_max,
        epoch_seed,
    )
    synchronize()
    flux
end

"""Load the HDF5 file on CPU and copy all model arrays to the GPU."""
function load_disco_params_device(h5_path::AbstractString)
    isfile(h5_path) || error("DISCO HDF5 file not found: $h5_path")
    load_disco_fe5250(h5_path)
end

"""Basic integration test; run with `julia --project=. gpu_synthesis_eclipse_disco.jl test file.h5`."""
function test_disco_gpu(h5_path::AbstractString)
    # CUDA.functional || error("CUDA is not functional on this machine")
    p_host = DISCOParams(h5_path)
    @assert p_host.n_wave == length(p_host.wavelength)
    p = to_device(p_host)

    mu = CuArray(Float32[0.2, 0.5, 0.9])
    rv = CUDA.zeros(Float32, 3)
    wt = CuArray(Float32[1f0, 1f0, 1f0])
    lo, hi = extrema(p_host.wavelength)
    flux = synthesize_eclipse_disco(mu, rv, wt, p;
        wav_min=lo, wav_max=hi, n_wave=p_host.n_wave)
    synchronize()
    out = Array(flux)
    @assert all(isfinite, out)
    @assert any(!=(0f0), out)
    println("DISCO GPU test passed: n_wave=$(length(out)), flux range=$(extrema(out))")
    return out
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
   length(ARGS) >= 2 &&
   ARGS[1] == "test"
    test_disco_gpu(ARGS[2])
end
