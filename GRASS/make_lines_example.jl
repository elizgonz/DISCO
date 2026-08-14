using CUDA
include("src/disco_fe5250.jl")
using .DISCOFe5250
using PyPlot

# 1D GPU Kernel helper
function kernel_single_profile!(prof_d, p_device, mu, patch_id, epoch_seed)
    w = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if w <= length(prof_d)
        prof_d[w] = DISCOFe5250.disco_intensity(p_device, mu, Int32(w), patch_id, epoch_seed)
    end
    return nothing
end

function get_disco_profile_gpu(p_device::DISCOParamsDevice, mu::Float32; 
                                patch_id::Int32=Int32(1), epoch_seed::UInt32=UInt32(0x12345678))
    n_wave = Int(p_device.n_wave)
    prof_d = CUDA.zeros(Float32, n_wave)
    
    threads = 256
    blocks = cld(n_wave, threads)
    
    @cuda threads=threads blocks=blocks kernel_single_profile!(prof_d, p_device, mu, patch_id, epoch_seed)
    CUDA.synchronize()
    
    return Array(prof_d) # Single clean DtoH copy
end

# --- Test Execution ---
h5_path = "/storage/home/efg5335/work/sw/DISCO/GRASS/disco_fe5250_params.h5"
p_host = DISCOParams(h5_path)
p_device = to_device(p_host)

for mu in [1.0f0, 0.8f0, 0.5f0, 0.2f0]
    prof_julia = get_disco_profile_gpu(p_device, mu)
    plt.plot(p_host.wavelength, prof_julia, label = mu)

end
plt.legend()
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.title("DISCO-GRASS")
plt.savefig("C2L")
