#!/bin/sh

sbatch --job-name="TiP-TimePre-1.0-60" --output Logs/TiP-TimePre-1.0-60.txt Time_Tucker_parametric.sh indianPines 1.0 60
sbatch --job-name="TuP-TimePre-1.0-60" --output Logs/TuP-TimePre-1.0-60.txt Time_Tucker_parametric.sh paviaU 1.0 60
sbatch --job-name="TsA-TimePre-1.0-60" --output Logs/TsA-TimePre-1.0-60.txt Time_Tucker_parametric.sh salinas 1.0 60