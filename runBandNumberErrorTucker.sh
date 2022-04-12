#!/bin/sh

sbatch --job-name="TiP-BandError-1.0-60" --output Logs/TiP-BandError-1.0-60.txt Band_number_error_Tucker_parametric.sh indianPines 1.0 60
sbatch --job-name="TuP-BandError-1.0-60" --output Logs/TuP-BandError-1.0-60.txt Band_number_error_Tucker_parametric.sh paviaU 1.0 60
sbatch --job-name="TsA-BandError-1.0-60" --output Logs/TsA-BandError-1.0-60.txt Band_number_error_Tucker_parametric.sh salinas 1.0 60