#!/bin/bash
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_gpn_msa
#SBATCH --output=logs/gpn/%x_%j.out
#SBATCH --error=logs/gpn/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"
module load Python/3.11.3-GCCcore-12.3.0
source ".venv/bin/activate"
python3 models/gpn-msa.py "$@"

echo "Done!"
