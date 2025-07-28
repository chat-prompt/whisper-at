#!/bin/bash
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="w-as-high"
#SBATCH --output=./log/%j_as.txt

set -euo pipefail  # Exit on error, undefined variables, pipe failures
set -x

# Script configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Create log directory if it doesn't exist
mkdir -p "./log"

# Environment setup
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Running in SLURM environment (Job ID: ${SLURM_JOB_ID})"
else
    echo "Running in local environment"
fi

# Activate virtual environment
# shellcheck source=/dev/null
VENV_PATH=$(poetry env info --path 2>/dev/null)
if [[ -n "$VENV_PATH" && -f "$VENV_PATH/bin/activate" ]]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Poetry virtual environment not found or not activated"
    echo "Please run 'poetry install' and 'poetry shell' first"
    exit 1
fi

# Set TORCH_HOME relative to project root
export TORCH_HOME="${PROJECT_ROOT}/pretrained_models"

# Training hyperparameters
lr=1e-6
freqm=0
timem=10
mixup=0.5
batch_size=48
model=whisper-high-lw_tr_1_8  # Options: whisper-high-lw_tr_1_8 (tl-tr), whisper-high-lw_down_tr_512_1_8 (tl-tr-512)
model_size=large-v1

# Dataset configuration
dataset=audioset_sonyc
bal=none
epoch=50
weight_decay=1e-5

# Learning rate scheduler
lrscheduler_start=15
lrscheduler_decay=0.75
lrscheduler_step=5

# Weight averaging
wa=True
wa_start=36
wa_end=50

# Learning rate adaptation
lr_adapt=True
lr_patience=2

# Data paths - use absolute paths for SLURM compatibility
DATA_DIR="/mnt/ssd_disk/github/whisper-at/data/processed_data"
tr_data="${DATA_DIR}/combined_train.json"
te_data="${DATA_DIR}/combined_val.json"
label_csv="${DATA_DIR}/class_labels_indices_extended.csv"
n_class=533
label_smooth=0.1

# Pretrained model path
pretrained_model="/mnt/ssd_disk/github/whisper-at/pretrained_models/large-v1_ori.pth"

# Verify required files exist
for file in "$tr_data" "$te_data" "$label_csv" "$pretrained_model"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: Required file not found: $file"
        exit 1
    fi
done

# Get current timestamp in YYMMDDHHMM format
timestamp=$(date +%y%m%d%H%M)

# Create experiment directory
exp_dir="${SCRIPT_DIR}/exp/combined-ft-${dataset}-${model}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-ep${epoch}-bs${batch_size}-lda${lr_adapt}-ls${label_smooth}-mix${mixup}-${freqm}-${timem}-${timestamp}"
mkdir -p "$exp_dir"

# Log experiment configuration
cat > "${exp_dir}/config.txt" <<EOF
Experiment Configuration
========================
Model: ${model}
Model Size: ${model_size}
Dataset: ${dataset}
Learning Rate: ${lr}
Batch Size: ${batch_size}
Epochs: ${epoch}
Number of Classes: ${n_class}
Label Smoothing: ${label_smooth}
Mixup: ${mixup}
Weight Decay: ${weight_decay}
LR Scheduler Start: ${lrscheduler_start}
LR Scheduler Decay: ${lrscheduler_decay}
LR Scheduler Step: ${lrscheduler_step}
Weight Averaging: ${wa} (${wa_start}-${wa_end})
LR Adaptation: ${lr_adapt} (patience: ${lr_patience})
Timestamp: ${timestamp}
EOF

echo "Starting training with configuration saved to: ${exp_dir}/config.txt"

# Run training
set +e  # Temporarily disable strict error handling
python -W ignore "${SCRIPT_DIR}/run.py" \
  --model "${model}" \
  --dataset "${dataset}" \
  --data-train "${tr_data}" \
  --data-val "${te_data}" \
  --exp-dir "${exp_dir}" \
  --label-csv "${label_csv}" \
  --n_class "${n_class}" \
  --lr "${lr}" \
  --n-epochs "${epoch}" \
  --batch-size "${batch_size}" \
  --save_model True \
  --freqm "${freqm}" \
  --timem "${timem}" \
  --mixup "${mixup}" \
  --bal "${bal}" \
  --model_size "${model_size}" \
  --label_smooth "${label_smooth}" \
  --lrscheduler_start "${lrscheduler_start}" \
  --lrscheduler_decay "${lrscheduler_decay}" \
  --lrscheduler_step "${lrscheduler_step}" \
  --loss BCE \
  --metrics mAP \
  --warmup True \
  --wa "${wa}" \
  --wa_start "${wa_start}" \
  --wa_end "${wa_end}" \
  --lr_adapt "${lr_adapt}" \
  --lr_patience "${lr_patience}" \
  --num-workers 8 \
  --pretrained_model "${pretrained_model}" \
  --weight_decay "${weight_decay}"
train_exit_code=$?
set -e  # Re-enable strict error handling

# Check if training completed successfully
if [ $train_exit_code -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved to: ${exp_dir}"
else
    echo "Training failed with exit code: ${train_exit_code}"
    exit $train_exit_code
fi