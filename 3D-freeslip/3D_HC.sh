#!/bin/bash
#PBS -q normal
#PBS -P nm33
#PBS -l ncpus=2064
#PBS -l mem=8200GB
#PBS -l jobfs=100GB
#PBS -l walltime=5:00:00
#PBS -l software=python
#PBS -l wd
#PBS -N 3d5e9fs-st
#PBS -W umask=0022

# Load modules.
module load fftw3
module use /g/data/hh5/public/modules
module load conda/analysis3-unstable

# Run Python applications
mpirun -n 2048 python3 3D_HC.py > $PBS_JOBID.log
