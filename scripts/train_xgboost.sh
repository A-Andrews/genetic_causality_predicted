#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=train_xgboost
#SBATCH --output=logs/xgboost/%x_%j.out
#SBATCH --error=logs/xgboost/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"
module load Python/3.11.3-GCCcore-12.3.0
source ".venv/bin/activate"
python3 models/run_xgboost.py "$@"

echo "Done!"
