#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=imagnet
#SBATCH --output=logs/imagnet_notexttoken.feit
#SBATCH --account=punim0619
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=feit-gpu-a100
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=feit

# check that the script is launched with sbatch #SBATCH feit-gpu-a100  gpu-h100 #SBATCH --qos=feit
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from this directory:
cd /data/gpfs/projects/punim0619/jiaming/MAPLE

# Activate environment
# Initialize conda for bash shell
source ~/.bashrc
# source activate llm-factory
module load anaconda
conda activate jm

# The modules to load:
module load cuDNN/8.9.3.28-CUDA-12.2.0
module load NCCL/2.18.3-CUDA-12.2.0

nvidia-smi

# Function to find an available port
#function find_free_port {
#    while true; do
#        port=$(shuf -i 20000-30000 -n 1)
#        if ! ss -lntu | grep -q ":$port "; then
#            echo $port
#            return
#        fi
#    done
#}

#  'checkpoints/fund_continue_104501.pt' 'scratch' 'checkpoints/fund.pt' 'checkpoints/fund_continue_104501.pt' 'scratch'
# '/data/projects/punim0619/datasets/coco2014/' /data/gpfs/projects/punim0619/jiaming/flickr30k/images/
# 'coco_retrieval' 'flickr30k' 'coco_caption' snli_ve "ImprovedContrastiveLoss"  'scratch'  "Cosine"

python train.py --root='/data/gpfs/projects/punim0619/jiaming/datasets'



