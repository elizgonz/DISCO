module DISCOFe5250

using CUDA
using HDF5
using CUDA.Adapt

export DISCOParams, DISCOParamsDevice, to_device, load_disco_fe5250
export disco_component_value, disco_intensity

const COMP_NAMES = ("GT", "OGR", "IgL")

struct DISCOParams
    wavelength::Vector{Float32}
    mu_min::Float32
    mu_max::Float32
    n_mu::Int
    n_wave::Int
    n_pca::Int
    mean_profiles::NTuple{3,Vector{Float32}}
    eigenprofiles::NTuple{3,Matrix{Float32}}
    pca_coeff_grids::NTuple{3,Matrix{Float32}}
    e1_alpha::Vector{Float32}
    e1_zeta::Vector{Float32}
    e1_omega::Vector{Float32}
    e1_med::Vector{Float32}
    e2_alpha::Vector{Float32}
    e2_zeta::Vector{Float32}
    e2_omega::Vector{Float32}
    e2_med::Vector{Float32}
end

"""Read the HDF5 file produced by export_disco_params.py."""
function DISCOParams(path::AbstractString)
    h5open(path, "r") do f
        a = attrs(f)
        wavelength = Float32.(read(f["wavelength"]))
        mu_min = Float32(a["mu_min"])
        mu_max = Float32(a["mu_max"])
        n_mu = Int(a["n_mu"])
        n_wave = Int(a["n_wave"])
        n_pca = Int(a["n_pca"])

        means = ntuple(3) do i
            Float32.(read(f["pca/$(COMP_NAMES[i])/mean_profile"]))
        end

        # Ensure eigenprofiles is (n_pca, n_wave) in Julia
        phis = ntuple(3) do i
            raw = Float32.(read(f["pca/$(COMP_NAMES[i])/eigenprofiles"]))
            if size(raw) == (n_pca, n_wave)
                raw
            elseif size(raw) == (n_wave, n_pca)
                permutedims(raw)
            else
                error("Unexpected eigenprofiles size for $(COMP_NAMES[i]): $(size(raw))")
            end
        end

        # Ensure pca_coeff_grid is (n_pca, n_mu) in Julia
        grids = ntuple(3) do i
            raw = Float32.(read(f["pca/$(COMP_NAMES[i])/pca_coeff_grid"]))
            if size(raw) == (n_pca, n_mu)
                raw
            elseif size(raw) == (n_mu, n_pca)
                permutedims(raw)
            else
                error("Unexpected PCA grid size for $(COMP_NAMES[i]): $(size(raw))")
            end
        end

        d = f["distributions"]
        e1a = Float32.(read(d["epsilon1/alpha"]))
        e1z = Float32.(read(d["epsilon1/zeta"]))
        e1w = Float32.(read(d["epsilon1/omega"]))
        e1m = Float32.(read(d["epsilon1/median"]))

        e2a = Float32.(read(d["epsilon2/alpha"]))
        e2z = Float32.(read(d["epsilon2/zeta"]))
        e2w = Float32.(read(d["epsilon2/omega"]))
        e2m = Float32.(read(d["epsilon2/median"]))

        @assert length(wavelength) == n_wave
        @assert length(e1a) == n_mu && length(e2a) == n_mu
        @assert size(phis[1]) == (n_pca, n_wave)
        @assert size(grids[1]) == (n_pca, n_mu)

        DISCOParams(wavelength, mu_min, mu_max, n_mu, n_wave, n_pca,
            means, phis, grids, e1a, e1z, e1w, e1m, e2a, e2z, e2w, e2m)
    end
end

"""Load the HDF5 model and immediately copy its arrays to the current GPU."""
load_disco_fe5250(path::AbstractString) = to_device(DISCOParams(path))

struct DISCOParamsDevice{
    V<:AbstractVector{Float32},
    M<:AbstractMatrix{Float32},
}
    wavelength::V
    mu_min::Float32
    mu_max::Float32
    n_mu::Int32
    n_wave::Int32
    n_pca::Int32

    mean_GT::V
    mean_OGR::V
    mean_IgL::V

    phi_GT::M
    phi_OGR::M
    phi_IgL::M

    a_GT::M
    a_OGR::M
    a_IgL::M

    e1_alpha::V
    e1_zeta::V
    e1_omega::V
    e1_med::V

    e2_alpha::V
    e2_zeta::V
    e2_omega::V
    e2_med::V
end

Adapt.@adapt_structure DISCOParamsDevice

function to_device(p::DISCOParams)
    c(x) = CuArray(x)
    DISCOParamsDevice(
        c(p.wavelength), p.mu_min, p.mu_max, Int32(p.n_mu),
        Int32(p.n_wave), Int32(p.n_pca),
        c(p.mean_profiles[1]), c(p.mean_profiles[2]), c(p.mean_profiles[3]),
        c(p.eigenprofiles[1]), c(p.eigenprofiles[2]), c(p.eigenprofiles[3]),
        c(p.pca_coeff_grids[1]), c(p.pca_coeff_grids[2]), c(p.pca_coeff_grids[3]),
        c(p.e1_alpha), c(p.e1_zeta), c(p.e1_omega), c(p.e1_med),
        c(p.e2_alpha), c(p.e2_zeta), c(p.e2_omega), c(p.e2_med)
    )
end

@inline function mu_index(
    mu::Float32,
    lo::Float32,
    hi::Float32,
    n::Int32,
)
    x = ifelse(mu < lo, lo, ifelse(mu > hi, hi, mu))
    pos = (x - lo) / (hi - lo) * Float32(n - Int32(1))

    i = Int32(floor(pos))
    i = ifelse(i < Int32(0), Int32(0), i)
    i = ifelse(i > n - Int32(2), n - Int32(2), i)

    return i + Int32(1), pos - Float32(i)
end

@inline function lerp(a::Float32, b::Float32, t::Float32)
    a + t * (b - a)
end

@inline function interp(
    arr::V,
    i::Int32,
    t::Float32,
) where {V<:AbstractVector{Float32}}
    @inbounds begin
        a = arr[Int(i)]
        b = arr[Int(i + Int32(1))]
        a + t * (b - a)
    end
end

@inline function disco_fractions(
    p::DISCOParamsDevice,
    mu::Float32,
    patch_id::Int32,
    epoch_seed::UInt32,
)
    i, t = mu_index(mu, p.mu_min, p.mu_max, p.n_mu)

    # When epoch_seed == 0, evaluate exact deterministic median filling factors
    if epoch_seed == UInt32(0)
        e1 = interp(p.e1_med, i, t)
        e2 = interp(p.e2_med, i, t)
        return e1, e2
    end

    a1 = interp(p.e1_alpha, i, t)
    z1 = interp(p.e1_zeta,  i, t)
    w1 = interp(p.e1_omega, i, t)

    a2 = interp(p.e2_alpha, i, t)
    z2 = interp(p.e2_zeta,  i, t)
    w2 = interp(p.e2_omega, i, t)

    seed = UInt32(patch_id) * UInt32(0x9e3779b9) + epoch_seed

    e1 = skewnormal(a1, z1, w1, seed)
    e2 = skewnormal(a2, z2, w2, seed + UInt32(0x85ebca6b))

    e1 = ifelse(e1 < 0f0, 0f0, ifelse(e1 > 1f0, 1f0, e1))
    e2 = ifelse(e2 < 0f0, 0f0, e2)

    return e1, e2
end

@inline function component_value(
    mean,
    phi,
    grid,
    i::Int32,
    t::Float32,
    n_pca::Int32,
    wave::Int32,
)
    @inbounds begin
        wi = Int(wave)
        ii = Int(i)

        v = mean[wi]
        k = Int32(1)

        while k <= n_pca
            coeff = lerp(
                grid[Int(k), ii],
                grid[Int(k), ii + 1],
                t,
            )
            v = muladd(coeff, phi[Int(k), wi], v)
            k += Int32(1)
        end

        return v
    end
end

@inline function disco_component_value(
    p::DISCOParamsDevice,
    mu::Float32,
    component::Int32,
    wave::Int32,
)
    i, t = mu_index(mu, p.mu_min, p.mu_max, p.n_mu)

    if component == Int32(1)
        return component_value(
            p.mean_GT, p.phi_GT, p.a_GT,
            i, t, p.n_pca, wave,
        )
    elseif component == Int32(2)
        return component_value(
            p.mean_OGR, p.phi_OGR, p.a_OGR,
            i, t, p.n_pca, wave,
        )
    else
        return component_value(
            p.mean_IgL, p.phi_IgL, p.a_IgL,
            i, t, p.n_pca, wave,
        )
    end
end

@inline function uniform01(x::UInt32)
    y = x ⊻ (x >> 16)
    y *= UInt32(0x7feb352d)
    y ⊻= y >> 15
    y *= UInt32(0x846ca68b)
    y ⊻= y >> 16
    (Float32(y) + 0.5f0) / 4294967296f0
end

@inline function skewnormal(
    alpha::Float32,
    zeta::Float32,
    omega::Float32,
    seed::UInt32,
)
    u0 = uniform01(seed)
    u0 = ifelse(u0 < 1f-7, 1f-7, u0)

    u1 = uniform01(seed + UInt32(0x9e3779b9))
    n0 = sqrt(-2f0 * log(u0)) *
         cos(6.28318530718f0 * u1)

    u2 = uniform01(seed + UInt32(0x243f6a88))
    u2 = ifelse(u2 < 1f-7, 1f-7, u2)

    u3 = uniform01(seed + UInt32(0xb7e15162))
    n1 = sqrt(-2f0 * log(u2)) *
         cos(6.28318530718f0 * u3)

    delta = alpha / sqrt(1f0 + alpha * alpha)
    rem = 1f0 - delta * delta
    rem = ifelse(rem < 0f0, 0f0, rem)

    return zeta + omega * (
        delta * abs(n0) + sqrt(rem) * n1
    )
end

@inline function disco_intensity(
    p::DISCOParamsDevice,
    mu::Float32,
    wave::Int32,
    patch_id::Int32,
    epoch_seed::UInt32,
)
    e1, e2 = disco_fractions(p, mu, patch_id, epoch_seed)

    gt  = disco_component_value(p, mu, Int32(1), wave)
    ogr = disco_component_value(p, mu, Int32(2), wave)
    igl = disco_component_value(p, mu, Int32(3), wave)

    one_minus_e1 = 1f0 - e1
    denom = 1f0 + e2

    return gt * e1 +
           ogr * (e2 * one_minus_e1 / denom) +
           igl * (one_minus_e1 / denom)
end

end

