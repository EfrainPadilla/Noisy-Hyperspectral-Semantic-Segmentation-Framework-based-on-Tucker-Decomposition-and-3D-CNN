#!/bin/sh
#SBATCH -p HPC
#SBATCH --output Logs/VBMF-Analysis-Results.txt
#SBATCH --cpus-per-task=32
#SBATCH --mem=124000
#SBATCH --exclusive=mcs
#SBATCH --job-name="VBMF-Analysis"

srun python3 vbmfAnalysis.py indianPines alpha-1.0 60dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 55dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 50dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 45dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 40dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 35dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 30dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 25dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 20dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 15dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 10dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 5dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 0dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 -5dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 -10dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 -15dB
srun python3 vbmfAnalysis.py indianPines alpha-1.0 -20dB

srun python3 vbmfAnalysis.py paviaU alpha-1.0 60dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 55dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 50dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 45dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 40dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 35dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 30dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 25dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 20dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 15dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 10dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 5dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 0dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 -5dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 -10dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 -15dB
srun python3 vbmfAnalysis.py paviaU alpha-1.0 -20dB
