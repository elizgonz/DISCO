#!/bin/bash
#SBATCH -A dfc13_mri
#SBATCH -p mgc-mri
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --time=48:00:00
#SBATCH --job-name=disco
#SBATCH --chdir=/storage/home/efg5335/work/sw/DISCO/GRASS
#SBATCH --output=/storage/home/efg5335/work/sw/DISCO/GRASS/disco.out

echo "========================================================"
echo "Starting Slurm Job: $SLURM_JOB_NAME (ID: $SLURM_JOBID)"
echo "Node: $SLURMD_NODENAME"
date
echo "========================================================"

# 1. Clear system library path pollutions and load CUDA 12
module purge 2>/dev/null || true
module load cuda/12.2 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda/12.0 2>/dev/null || true

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 2. GPU Diagnostics
echo "--- GPU Info ---"
nvidia-smi || echo "WARNING: nvidia-smi failed or no GPU found"

WORKDIR="/storage/home/efg5335/work/sw/DISCO/GRASS"
H5_PATH="${WORKDIR}/disco_fe5250_params.h5"

cd "${WORKDIR}"
echo "Working directory: $(pwd)"

# 3. Pin CUDA runtime version to 12.2 for CUDA.jl and Volta V100 (sm_70) compatibility
echo "--- Setting CUDA runtime to 12.2.0 ---"
julia --project=./julia_env -e '
using CUDA
try
    CUDA.set_runtime_version!(v"12.2.0")
    println("Successfully set CUDA runtime version to 12.2.0")
catch e
    println("CUDA set_runtime_version note: ", e)
end
'

# 4. Execute test
echo "--- Starting Julia DISCO GPU Synthesis Test ---"
julia --project=./julia_env \
    src/gpu_synthesis_eclipse_disco.jl test \
    "${H5_PATH}"

EXIT_CODE=$?

echo "========================================================"
echo "Julia exited with code: ${EXIT_CODE}"
date
echo "========================================================"
