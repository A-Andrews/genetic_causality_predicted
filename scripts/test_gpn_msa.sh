#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --job-name=gpn_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
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
python3 gpn_test/gpn_test.py --msa /well/palamara/users/nrw600/contribution_prediction/gpn-msa_data/89.zarr --device cuda:0

echo "Done!"
