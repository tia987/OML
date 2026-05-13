#!/bin/bash
#SBATCH --job-name=0ML26            # Job name    (default: sbatch)
#SBATCH --output=oml.out            # Output file (default: slurm-%j.out)
#SBATCH --error=oml.err             # Error file  (default: slurm-%j.out)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --constraint=EPYC_7763      # Select node with CPU
#SBATCH --mem-per-cpu=1024          # Memory per CPU
#SBATCH --time=24:00:00             # Wall clock time limit
#SBATCH --mail-type=END,FAIL        # Send an email when job ends

# Load some modules
module load stack/2025-06 gcc/12.2.0 python/3.13.0 eth_proxy
module list

# Install required packages (source ./installation.sh)
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt

# Go into folder
cd OptML_CNN_project

# Run the program
python3 main.py -n_trials=20