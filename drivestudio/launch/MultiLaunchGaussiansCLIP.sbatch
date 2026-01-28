#!/bin/bash -l
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --mail-type=END
#SBATCH --array=0-29%5
#SBATCH --job-name=streetgsGaussiansCLIP
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# (Optional) load site modules if needed, e.g.:
# module load apptainer
# module load cuda

# Baseline GPU check
/usr/bin/nvidia-smi || true

previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null | /usr/bin/tail -n '+2' || true)

# Paths
cd /tudelft.net/staff-umbrella/hchassagnette/Workspace
IMG=/tudelft.net/staff-umbrella/hchassagnette/Workspace/drivestudio.sif

# ---- Define your 10 scene indices here (edit to match what you want) ----
SCENES=(000 001 002 003 004 005 006 007 008 009)
CONFIGS=(
/tudelft.net/staff-umbrella/hchassagnette/Workspace/3DGS.Thesis/drivestudio/configs/semantictest/clip/streetgsSemantic_CLIP_0,001_Entropy_0.0003_Focal_0,001_N2.yaml
/tudelft.net/staff-umbrella/hchassagnette/Workspace/3DGS.Thesis/drivestudio/configs/semantictest/clip/streetgsSemantic_CLIP_0,0001_Entropy_0.0003_Focal_0,001_N2.yaml
/tudelft.net/staff-umbrella/hchassagnette/Workspace/3DGS.Thesis/drivestudio/configs/semantictest/clip/streetgsSemantic_CLIP_0,0003_Entropy_0.0003_Focal_0,001_N2.yaml
)

num_scenes=${#SCENES[@]}
num_configs=${#CONFIGS[@]}
task=${SLURM_ARRAY_TASK_ID}

# Map linear task id -> (config_idx, scene_idx)
scene_i=$(( task % num_scenes ))
config_i=$(( task / num_scenes ))

scene_idx="${SCENES[$scene_i]}"
config_file="${CONFIGS[$config_i]}"

# Safety check (if you later change --array size)
if (( config_i >= num_configs )); then
  echo "Invalid task id $task (config_i=$config_i >= $num_configs). Exiting."
  exit 0
fi

# Nice run naming: include config “tag” + scene
cfg_tag="$(basename "$config_file")"
cfg_tag="${cfg_tag%.yaml}"               # streetgs, streetgs_ablation, ...
run_name="${cfg_tag}_${scene_idx}"       # e.g. streetgs_000

echo "task=$task config=$config_file scene=$scene_idx run_name=$run_name"

# Run inside the container
srun apptainer exec --nv "$IMG" bash -lc "
  cd /tudelft.net/staff-umbrella/hchassagnette/Workspace/3DGS.Thesis/drivestudio
  set -euo pipefail
  source /home/miniconda3/bin/activate
  conda activate drivestudio || true

  export PYTHONPATH=/tudelft.net/staff-umbrella/hchassagnette/Workspace/3DGS.Thesis/drivestudio
  python tools/train.py \
    --config_file ${config_file} \
    --output_root /tudelft.net/staff-umbrella/hchassagnette/Workspace/output \
    --project ThesisResults \
    --run_name ${run_name} \
    dataset=nuscenes/6cams \
    data.scene_idx=${scene_idx} \
    data.start_timestep=0 \
    data.end_timestep=-1
"

# GPU usage delta without failing the job
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null \
  | /usr/bin/grep -v -F "$previous" || true

