#!/bin/bash
#PBS -q normal 
#PBS -l ncpus=256
#PBS -l mem=256GB
#PBS -l jobfs=400GB
#PBS -l walltime=24:00:00
#PBS -l software=python
#PBS -l wd
 
# Load modules.
module load python3/3.6.2
module load mpi4py/3.0.0-py36-omp10.2
module load hdf5/1.8.14

# Run Python applications
mpirun -n 256 python3 MagneticZI-poloidalB0-drag.py > $PBS_JOBID.log   
