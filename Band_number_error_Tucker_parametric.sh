#!/bin/sh
#SBATCH -p HPC
#SBATCH --cpus-per-task=4
#SBATCH --mem=102400
#SBATCH --exclusive=mcs
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:59
#SBATCH --gres=gpu:1

srun --gres=gpu:1 python3 hsi_tucker_band_number_error_ep.py ${1} alpha-${2} ${3}dB
