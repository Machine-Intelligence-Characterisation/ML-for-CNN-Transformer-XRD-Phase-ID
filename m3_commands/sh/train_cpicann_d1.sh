#!/bin/bash
#SBATCH --job-name=cpicann_d1_train
#SBATCH --output=/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/monash_HPC_commands/slurm_outputs/cpicann_d1_train_%j.out
#SBATCH --error=/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/monash_HPC_commands/slurm_outputs/cpicann_d1_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# Create output directory if it doesn't exist
mkdir -p /home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/monash_HPC_commands/slurm_outputs

# Load modules
module load intel-python/2024.1.0
module load cuda/12.2.0

# Navigate to project directory
cd /home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase

# Install required packages if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm pandas tensorboardX

# Run CPICANN training
python src/training/train_cpicann_d1.py \
    --epochs 100 \
    --warmup-epochs 10 \
    --lr 8e-5 \
    --batch_size 64 \
    --embed_dim 128 \
    --num_classes 23073 \
    --data_dir_train training_data/D1_full/train/ \
    --data_dir_val training_data/D1_full/val/ \
    --anno_train training_data/D1_full/anno_train.csv \
    --anno_val training_data/D1_full/anno_val.csv \
    --output_dir trained_models/ \
    --model_name CPICANN_D1_full \
    --device cuda:0

echo "CPICANN training completed!"
