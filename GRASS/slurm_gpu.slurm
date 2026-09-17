#!/bin/bash
#SBATCH -A dfc13_mri
#SBATCH -p mgc-mri
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --time=48:00:00
#SBATCH --job-name=DISCO_GRASS_RV
#SBATCH --chdir=/storage/home/efg5335/work/sw/DISCO/GRASS/
#SBATCH --output=/storage/home/efg5335/work/sw/DISCO/GRASS/make_lines_example.out

echo "About to start: $SLURM_JOB_NAME"
date
echo "Job id: $SLURM_JOBID"
echo "About to change into $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR
echo "About to start Julia"
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
julia --project=./julia_env make_lines_comp.jl
echo "Julia exited"
date