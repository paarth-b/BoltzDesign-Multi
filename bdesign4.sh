#!/bin/bash
#SBATCH --job-name=boltz_design_fgfr2_3b
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 4-00:00:00           # time in d-hh:mm:s
#SBATCH -p general                # partition
#SBATCH -q public            # QOS
#SBATCH -G a100:1
#SBATCH --mem=40G
#SBATCH -o sl.%j.out         # file to save job's STDOUT (%j = JobId)
#SBATCH -e sl.%j.err         # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE           # Purge the job-submitting shell environmet

module purge

module load mamba/latest

source activate boltz_design

cd /scratch/phjiang/abhi_scratc/setup/BoltzDesign1

~/.conda/envs/boltz_design/bin/python boltzdesign1.py \
  --target_name fgfr2_3b_599 \
  --target_type protein \
  --pdb_path ./frame_599.pdb \
  --pdb_target_ids X \
  --constraint_target X \
  --use_msa True \
  --length_min 50 \
  --length_max 100 \
  --optimizer_type AdamW \
  --gpu_id 0 \
  --design_samples 5 \
  --num_designs 16 \
  --save_trajectory True \
  --run_alphafold False \
  --num_inter_contacts 3 \
  --helix_loss_max -0.39 \
  --helix_loss_min -1 \
  --run_rosetta True \
  --redo_boltz_predict True \
  --show_animation False


conda deactivate
#  --contact_residues "137,138,139,140,141,142,143,144,145,146,147,148,151,152,153,154,156,158,186,188,191,193,195,197,220,221,222,223,225" \
