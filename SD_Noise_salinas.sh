#!/bin/sh
#SBATCH --output Logs/SDNoise-salinas.txt
#SBATCH -p HPC
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --job-name="salNoiAdd"
#SBATCH --exclusive=mcs
#SBATCH --time=01:59:59

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 -20
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 -20
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 -20
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 -20
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 -20
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 -20
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 -20
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 -20
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 -20
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 -20
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 -20

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 -15
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 -15
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 -15
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 -15
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 -15
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 -15
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 -15
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 -15
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 -15
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 -15
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 -15

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 -10
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 -10
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 -10
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 -10
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 -10
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 -10
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 -10
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 -10
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 -10
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 -10
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 -10

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 -5
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 -5
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 -5
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 -5
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 -5
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 -5
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 -5
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 -5
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 -5
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 -5
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 -5

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 0
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 0
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 0
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 0
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 0
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 0
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 0
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 0
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 0
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 0
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 0

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 5
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 5
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 5
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 5
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 5
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 5
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 5
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 5
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 5
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 5
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 5

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 10
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 10
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 10
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 10
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 10
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 10
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 10
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 10
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 10
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 10
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 10

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 15
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 15
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 15
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 15
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 15
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 15
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 15
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 15
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 15
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 15
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 15

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 20
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 20
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 20
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 20
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 20
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 20
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 20
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 20
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 20
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 20
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 20

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 25
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 25
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 25
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 25
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 25
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 25
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 25
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 25
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 25
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 25
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 25

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 30
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 30
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 30
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 30
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 30
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 30
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 30
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 30
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 30
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 30
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 30

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 35
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 35
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 35
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 35
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 35
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 35
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 35
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 35
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 35
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 35
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 35

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 40
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 40
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 40
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 40
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 40
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 40
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 40
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 40
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 40
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 40
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 40

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 45
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 45
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 45
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 45
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 45
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 45
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 45
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 45
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 45
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 45
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 45

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 50
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 50
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 50
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 50
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 50
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 50
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 50
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 50
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 50
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 50
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 50

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 55
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 55
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 55
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 55
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 55
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 55
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 55
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 55
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 55
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 55
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 55

srun python3 hsi_dependent_noiseadd_ep.py salinas 0.1 60
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.2 60
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.25 60
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.33 60
srun python3 hsi_dependent_noiseadd_ep.py salinas 0.5 60
srun python3 hsi_dependent_noiseadd_ep.py salinas 1.0 60
srun python3 hsi_dependent_noiseadd_ep.py salinas 2.0 60
srun python3 hsi_dependent_noiseadd_ep.py salinas 3.0 60
srun python3 hsi_dependent_noiseadd_ep.py salinas 4.0 60
srun python3 hsi_dependent_noiseadd_ep.py salinas 5.0 60
srun python3 hsi_dependent_noiseadd_ep.py salinas 10.0 60
