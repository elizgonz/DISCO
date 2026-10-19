module DISCOJulia

using CUDA
using HDF5
using CUDA.Adapt

export DISCOParams, DISCOParamsDevice, to_device, load_disco
export disco_component_value, disco_intensity, disco_intensity_original
export original_cb_offset, cb_polynomial

const COMP_NAMES = ("GT", "OGR", "IgL")
const REFERENCE_WAVELENGTHS_ANGSTROM = Dict(
    "Fe5250" => 5250.2084,
    "Fe6152" => 6151.6170,
    "Fe6173" => 6173.3344,
)

function require_shape(name, values, expected)
    size(values) == expected ||
        error("Unexpected $(name) size: expected $(expected), got $(size(values))")
    all(isfinite, values) || error("$(name) contains non-finite values")
    return values
end

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
    original_cb_offsets_nm::Vector{Float32}
    cb_poly_c0_nm::Float32
    cb_poly_c1_nm_per_mu::Float32
    cb_poly_c2_nm_per_mu2::Float32
end

function DISCOParams(
    wavelength::Vector{Float32}, mu_min::Float32, mu_max::Float32,
    n_mu::Int, n_wave::Int, n_pca::Int,
    mean_profiles::NTuple{3,Vector{Float32}},
    eigenprofiles::NTuple{3,Matrix{Float32}},
    pca_coeff_grids::NTuple{3,Matrix{Float32}},
    e1_alpha::Vector{Float32}, e1_zeta::Vector{Float32},
    e1_omega::Vector{Float32}, e1_med::Vector{Float32},
    e2_alpha::Vector{Float32}, e2_zeta::Vector{Float32},
    e2_omega::Vector{Float32}, e2_med::Vector{Float32},
)
    DISCOParams(
        wavelength, mu_min, mu_max, n_mu, n_wave, n_pca,
        mean_profiles, eigenprofiles, pca_coeff_grids,
        e1_alpha, e1_zeta, e1_omega, e1_med,
        e2_alpha, e2_zeta, e2_omega, e2_med,
        zeros(Float32, n_mu), 0f0, 0f0, 0f0,
    )
end

"""Read and validate the HDF5 file produced by export_params.py."""
function DISCOParams(path::AbstractString)
    h5open(path, "r") do f
        a = attrs(f)
        wavelength = Float32.(read(f["wavelength"]))
        mu_min = Float32(a["mu_min"])
        mu_max = Float32(a["mu_max"])
        n_mu = Int(a["n_mu"])
        n_wave = Int(a["n_wave"])
        n_pca = Int(a["n_pca"])
        mu_grid = Float32.(read(f["mu_grid"]))

        n_mu >= 2 || error("n_mu must be at least 2")
        n_wave >= 2 || error("n_wave must be at least 2")
        n_pca >= 1 || error("n_pca must be positive")
        0f0 < mu_min < mu_max <= 1f0 ||
            error("Invalid mu range: [$(mu_min), $(mu_max)]")
        require_shape("mu_grid", mu_grid, (n_mu,))
        isapprox(mu_grid[1], mu_min; atol=eps(Float32)) ||
            error("mu_grid lower endpoint does not match mu_min")
        isapprox(mu_grid[end], mu_max; atol=eps(Float32)) ||
            error("mu_grid upper endpoint does not match mu_max")
        all(diff(mu_grid) .> 0f0) || error("mu_grid must be strictly increasing")
        expected_mu = collect(range(mu_min, mu_max; length=n_mu))
        maximum(abs.(mu_grid .- expected_mu)) <= 8f0 * eps(Float32) ||
            error("mu_grid must be uniformly spaced")
        require_shape("wavelength", wavelength, (n_wave,))
        all(diff(wavelength) .> 0f0) ||
            error("wavelength must be strictly increasing")

        means = ntuple(3) do i
            require_shape(
                "pca/$(COMP_NAMES[i])/mean_profile",
                Float32.(read(f["pca/$(COMP_NAMES[i])/mean_profile"])),
                (n_wave,),
            )
        end

        phis = ntuple(3) do i
            raw = Float32.(read(f["pca/$(COMP_NAMES[i])/eigenprofiles"]))
            phi = if size(raw) == (n_pca, n_wave)
                raw
            elseif size(raw) == (n_wave, n_pca)
                permutedims(raw)
            else
                error("Unexpected eigenprofiles size for $(COMP_NAMES[i]): $(size(raw))")
            end
            require_shape("pca/$(COMP_NAMES[i])/eigenprofiles", phi, (n_pca, n_wave))
        end

        grids = ntuple(3) do i
            raw = Float32.(read(f["pca/$(COMP_NAMES[i])/pca_coeff_grid"]))
            grid = if size(raw) == (n_pca, n_mu)
                raw
            elseif size(raw) == (n_mu, n_pca)
                permutedims(raw)
            else
                error("Unexpected PCA grid size for $(COMP_NAMES[i]): $(size(raw))")
            end
            require_shape("pca/$(COMP_NAMES[i])/pca_coeff_grid", grid, (n_pca, n_mu))
        end

        d = f["distributions"]
        distribution(name, field) = require_shape(
            "distributions/$(name)/$(field)",
            Float32.(read(d["$(name)/$(field)"])),
            (n_mu,),
        )
        e1a = distribution("epsilon1", "alpha")
        e1z = distribution("epsilon1", "zeta")
        e1w = distribution("epsilon1", "omega")
        e1m = distribution("epsilon1", "median")

        e2a = distribution("epsilon2", "alpha")
        e2z = distribution("epsilon2", "zeta")
        e2w = distribution("epsilon2", "omega")
        e2m = distribution("epsilon2", "median")

        params = DISCOParams(wavelength, mu_min, mu_max, n_mu, n_wave, n_pca,
            means, phis, grids, e1a, e1z, e1w, e1m, e2a, e2z, e2w, e2m)
        line = haskey(a, "line") ? String(a["line"]) : infer_line(wavelength)
        haskey(REFERENCE_WAVELENGTHS_ANGSTROM, line) ||
            error("No reference wavelength configured for DISCO line $(line)")
        reference_nm = REFERENCE_WAVELENGTHS_ANGSTROM[line] / 10
        offsets_nm = extract_core_offsets(params, mu_grid, reference_nm)
        coefficients = fit_cb_polynomial(mu_grid, offsets_nm)
        DISCOParams(
            wavelength, mu_min, mu_max, n_mu, n_wave, n_pca,
            means, phis, grids, e1a, e1z, e1w, e1m, e2a, e2z, e2w, e2m,
            Float32.(offsets_nm), Float32(coefficients[1]),
            Float32(coefficients[2]), Float32(coefficients[3]),
        )
    end
end

"""Load the HDF5 model and immediately copy its arrays to the current GPU."""
load_disco(path::AbstractString) = to_device(DISCOParams(path))

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
    original_cb_offsets_nm::V
    cb_poly_c0_nm::Float32
    cb_poly_c1_nm_per_mu::Float32
    cb_poly_c2_nm_per_mu2::Float32
end

Adapt.@adapt_structure DISCOParamsDevice

function DISCOParamsDevice(
    wavelength::V, mu_min::Float32, mu_max::Float32,
    n_mu::Int32, n_wave::Int32, n_pca::Int32,
    mean_GT::V, mean_OGR::V, mean_IgL::V,
    phi_GT::M, phi_OGR::M, phi_IgL::M,
    a_GT::M, a_OGR::M, a_IgL::M,
    e1_alpha::V, e1_zeta::V, e1_omega::V, e1_med::V,
    e2_alpha::V, e2_zeta::V, e2_omega::V, e2_med::V,
) where {V<:AbstractVector{Float32},M<:AbstractMatrix{Float32}}
    offsets = similar(wavelength, Int(n_mu))
    fill!(offsets, 0f0)
    DISCOParamsDevice(
        wavelength, mu_min, mu_max, n_mu, n_wave, n_pca,
        mean_GT, mean_OGR, mean_IgL, phi_GT, phi_OGR, phi_IgL,
        a_GT, a_OGR, a_IgL, e1_alpha, e1_zeta, e1_omega, e1_med,
        e2_alpha, e2_zeta, e2_omega, e2_med, offsets, 0f0, 0f0, 0f0,
    )
end

function to_device(p::DISCOParams)
    c(x) = CuArray(x)
    DISCOParamsDevice(
        c(p.wavelength), p.mu_min, p.mu_max, Int32(p.n_mu),
        Int32(p.n_wave), Int32(p.n_pca),
        c(p.mean_profiles[1]), c(p.mean_profiles[2]), c(p.mean_profiles[3]),
        c(p.eigenprofiles[1]), c(p.eigenprofiles[2]), c(p.eigenprofiles[3]),
        c(p.pca_coeff_grids[1]), c(p.pca_coeff_grids[2]), c(p.pca_coeff_grids[3]),
        c(p.e1_alpha), c(p.e1_zeta), c(p.e1_omega), c(p.e1_med),
        c(p.e2_alpha), c(p.e2_zeta), c(p.e2_omega), c(p.e2_med),
        c(p.original_cb_offsets_nm), p.cb_poly_c0_nm,
        p.cb_poly_c1_nm_per_mu, p.cb_poly_c2_nm_per_mu2,
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

function infer_line(wavelength::Vector{Float32})
    matches = filter(collect(keys(REFERENCE_WAVELENGTHS_ANGSTROM))) do line
        reference_nm = REFERENCE_WAVELENGTHS_ANGSTROM[line] / 10
        wavelength[1] <= reference_nm <= wavelength[end]
    end
    length(matches) == 1 ||
        error("Cannot infer a unique DISCO line from the wavelength grid")
    return only(matches)
end

@inline function host_component_value(
    p::DISCOParams, mu::Float32, component::Int, wave::Int,
)
    i, t = mu_index(mu, p.mu_min, p.mu_max, Int32(p.n_mu))
    return component_value(
        p.mean_profiles[component], p.eigenprofiles[component],
        p.pca_coeff_grids[component], i, t, Int32(p.n_pca), Int32(wave),
    )
end

@inline function host_intensity_original(p::DISCOParams, mu::Float32, wave::Int)
    i, t = mu_index(mu, p.mu_min, p.mu_max, Int32(p.n_mu))
    e1 = interp(p.e1_med, i, t)
    e2 = interp(p.e2_med, i, t)
    gt = host_component_value(p, mu, 1, wave)
    ogr = host_component_value(p, mu, 2, wave)
    igl = host_component_value(p, mu, 3, wave)
    one_minus_e1 = 1f0 - e1
    denom = 1f0 + e2
    return gt * e1 +
           ogr * (e2 * one_minus_e1 / denom) +
           igl * (one_minus_e1 / denom)
end

function extract_core_offsets(
    p::DISCOParams, mu_grid::Vector{Float32}, reference_nm::Float64,
)
    p.wavelength[1] <= reference_nm <= p.wavelength[end] ||
        error("Reference wavelength lies outside the DISCO wavelength grid")

    first_wave = searchsortedfirst(p.wavelength, reference_nm - 0.01)
    last_wave = searchsortedlast(p.wavelength, reference_nm + 0.01)
    if last_wave - first_wave < 2
        center = clamp(searchsortedfirst(p.wavelength, reference_nm), 2, p.n_wave - 1)
        first_wave = center - 1
        last_wave = center + 1
    end

    offsets = Vector{Float64}(undef, length(mu_grid))
    for mu_idx in eachindex(mu_grid)
        mu = mu_grid[mu_idx]
        minimum_wave = first_wave
        minimum_intensity = host_intensity_original(p, mu, first_wave)
        for wave in (first_wave + 1):last_wave
            intensity = host_intensity_original(p, mu, wave)
            if intensity < minimum_intensity
                minimum_wave = wave
                minimum_intensity = intensity
            end
        end

        core_offset = Float64(p.wavelength[minimum_wave]) - reference_nm
        if first_wave < minimum_wave < last_wave
            x0 = Float64(p.wavelength[minimum_wave - 1])
            x1 = Float64(p.wavelength[minimum_wave])
            x2 = Float64(p.wavelength[minimum_wave + 1])
            y0 = Float64(host_intensity_original(p, mu, minimum_wave - 1))
            y1 = Float64(minimum_intensity)
            y2 = Float64(host_intensity_original(p, mu, minimum_wave + 1))
            h0 = x1 - x0
            h1 = x2 - x1
            slope0 = (y1 - y0) / h0
            slope1 = (y2 - y1) / h1
            quadratic = (slope1 - slope0) / (h0 + h1)
            if quadratic > 0
                delta = -(slope0 + quadratic * h0) / (2 * quadratic)
                if -h0 <= delta <= h1
                    core_offset += delta
                end
            end
        end
        offsets[mu_idx] = core_offset
    end
    return offsets
end

function fit_cb_polynomial(mu_grid::Vector{Float32}, offsets_nm::Vector{Float64})
    length(mu_grid) == length(offsets_nm) ||
        error("μ and CB offset arrays must have matching lengths")
    system = zeros(Float64, 3, 4)
    for (mu, offset) in zip(mu_grid, offsets_nm)
        x = Float64(mu)
        x2 = x * x
        system[1, 1] += 1
        system[1, 2] += x
        system[1, 3] += x2
        system[1, 4] += offset
        system[2, 2] += x2
        system[2, 3] += x2 * x
        system[2, 4] += x * offset
        system[3, 3] += x2 * x2
        system[3, 4] += x2 * offset
    end
    system[2, 1] = system[1, 2]
    system[3, 1] = system[1, 3]
    system[3, 2] = system[2, 3]

    for column in 1:3
        pivot = argmax(abs.(system[column:3, column])) + column - 1
        abs(system[pivot, column]) > eps(Float64) ||
            error("Cannot fit CB polynomial to degenerate μ samples")
        if pivot != column
            row = copy(system[column, :])
            system[column, :] = system[pivot, :]
            system[pivot, :] = row
        end
        system[column, :] ./= system[column, column]
        for row in 1:3
            row == column && continue
            factor = system[row, column]
            system[row, :] .-= factor .* system[column, :]
        end
    end

    coefficients = system[:, 4]
    all(isfinite, coefficients) || error("CB polynomial fit produced non-finite coefficients")
    return coefficients
end

@inline function original_cb_offset(p::DISCOParams, mu::Float32)
    i, t = mu_index(mu, p.mu_min, p.mu_max, Int32(p.n_mu))
    return interp(p.original_cb_offsets_nm, i, t)
end

@inline function original_cb_offset(p::DISCOParamsDevice, mu::Float32)
    i, t = mu_index(mu, p.mu_min, p.mu_max, p.n_mu)
    return interp(p.original_cb_offsets_nm, i, t)
end

@inline function cb_polynomial(p::DISCOParams, mu::Real)
    x = clamp(Float64(mu), 0.0, 1.0)
    return Float64(p.cb_poly_c0_nm) +
           Float64(p.cb_poly_c1_nm_per_mu) * x +
           Float64(p.cb_poly_c2_nm_per_mu2) * x^2
end

@inline function cb_polynomial(p::DISCOParamsDevice, mu::Float32)
    x = ifelse(mu < 0f0, 0f0, ifelse(mu > 1f0, 1f0, mu))
    return p.cb_poly_c0_nm +
           p.cb_poly_c1_nm_per_mu * x +
           p.cb_poly_c2_nm_per_mu2 * x * x
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
    original_offset = original_cb_offset(p, mu)
    target_offset = cb_polynomial(p, mu)
    wavelength_offset = original_offset - target_offset
    if wavelength_offset == 0f0
        return disco_intensity_original(p, mu, wave, patch_id, epoch_seed)
    end
    return disco_intensity_original_with_offset(
        p, mu, wave, wavelength_offset, patch_id, epoch_seed,
    )
end

@inline function disco_intensity_original_with_offset(
    p::DISCOParamsDevice,
    mu::Float32,
    wave::Int32,
    wavelength_offset::Float32,
    patch_id::Int32,
    epoch_seed::UInt32,
)
    n = Int32(p.n_wave)
    base_wavelength = p.wavelength[Int(wave)]
    if wavelength_offset <= p.wavelength[1] - base_wavelength
        return disco_intensity_original(p, mu, Int32(1), patch_id, epoch_seed)
    elseif wavelength_offset >= p.wavelength[Int(n)] - base_wavelength
        return disco_intensity_original(p, mu, n, patch_id, epoch_seed)
    end

    lo = Int32(1)
    hi = n
    while hi - lo > Int32(1)
        mid = (lo + hi) ÷ Int32(2)
        relative_wavelength = p.wavelength[Int(mid)] - base_wavelength
        if relative_wavelength <= wavelength_offset
            lo = mid
        else
            hi = mid
        end
    end

    lower_offset = p.wavelength[Int(lo)] - base_wavelength
    interval = p.wavelength[Int(lo + Int32(1))] - p.wavelength[Int(lo)]
    fraction = (wavelength_offset - lower_offset) / interval
    f0 = disco_intensity_original(p, mu, lo, patch_id, epoch_seed)
    f1 = disco_intensity_original(p, mu, lo + Int32(1), patch_id, epoch_seed)
    return f0 + fraction * (f1 - f0)
end

@inline function disco_intensity_original(
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
