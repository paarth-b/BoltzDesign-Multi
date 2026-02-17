#!/bin/bash
#SBATCH --job-name=boltz_design_fgfr2_3b_multigpu
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 1-00:00:00           # time in d-hh:mm:s
#SBATCH -p general                # partition
#SBATCH -q public            # QOS
#SBATCH --gres=gpu:v100:2
#SBATCH --constraint=v100_32
#SBATCH --mem=80G
#SBATCH -o sl.%j.out         # file to save job's STDOUT (%j = JobId)
#SBATCH -e sl.%j.err         # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE           # Purge the job-submitting shell environmet

export NO_AI_TRACKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

module purge

module load mamba/latest

source activate boltz_design

cd /home/pbatra6/code/github/BoltzDesign-Multi

~/.conda/envs/boltz_design/bin/python boltzdesign.py \
  --target_name fgfr2_3b_sm_msa \
  --target_type protein \
  --pdb_path ./data/frame_450.pdb:./data/frame_599.pdb:./data/frame_1698.pdb:./data/frame_2071.pdb \
  --pdb_target_ids X \
  --constraint_target X \
  --use_msa False \
  --msa_max_seqs 128 \
  --length_min 50 \
  --length_max 80 \
  --optimizer_type AdamW \
  --gpu_ids 0,1 \
  --design_samples 2 \
  --num_designs 4 \
  --save_trajectory False \
  --num_inter_contacts 3 \
  --helix_loss_max -0.39 \
  --helix_loss_min -1 \
  --run_rosetta False \
  --redo_boltz_predict False \
  --show_animation False


conda deactivate
