#!/bin/bash
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="w-as-high"
#SBATCH --output=./log/%j_as.txt

set -euo pipefail
set -x

# Error handler
error_handler() {
    echo "Error occurred in script at line $1" >&2
    exit 1
}
trap 'error_handler $LINENO' ERR

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Environment setup
if [ -z "${VIRTUAL_ENV:-}" ]; then
    # Try to detect Poetry environment
    if command -v poetry &> /dev/null; then
        echo "Activating Poetry environment..."
        cd "${PROJECT_ROOT}"
        poetry shell
    else
        echo "Warning: No virtual environment detected" >&2
    fi
fi

export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/pretrained_models}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Training parameters
lr="${LR:-1e-6}"
freqm="${FREQM:-0}"
timem="${TIMEM:-10}"
mixup="${MIXUP:-0.5}"
batch_size="${BATCH_SIZE:-48}"
model="${MODEL:-whisper-high-lw_tr_1_8}"
model_size="${MODEL_SIZE:-large-v1}"

# Dataset configuration
dataset="${DATASET:-audioset_sonyc}"
bal="${BALANCE:-none}"
epoch="${EPOCHS:-50}"
weight_decay="${WEIGHT_DECAY:-1e-5}"
lrscheduler_start="${LR_SCHEDULER_START:-15}"
lrscheduler_decay="${LR_SCHEDULER_DECAY:-0.75}"
lrscheduler_step="${LR_SCHEDULER_STEP:-5}"
wa="${WA:-True}"
wa_start="${WA_START:-36}"
wa_end="${WA_END:-50}"
lr_adapt="${LR_ADAPT:-True}"
lr_patience="${LR_PATIENCE:-2}"
label_smooth="${LABEL_SMOOTH:-0.1}"
n_class="${N_CLASS:-533}"

# Data paths
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/processed_data}"
tr_data="${TRAIN_DATA:-${DATA_DIR}/combined_train.json}"
te_data="${VAL_DATA:-${DATA_DIR}/combined_val.json}"
label_csv="${LABEL_CSV:-${DATA_DIR}/class_labels_indices_extended.csv}"

# Model path
PRETRAINED_DIR="${PRETRAINED_DIR:-${PROJECT_ROOT}/pretrained_models}"
pretrained_model="${PRETRAINED_MODEL:-${PRETRAINED_DIR}/large-v1_ori.pth}"

# Validate required files exist
required_files=(
    "${tr_data}"
    "${te_data}"
    "${label_csv}"
    "${pretrained_model}"
)

for file in "${required_files[@]}"; do
    if [ ! -f "${file}" ]; then
        echo "Error: Required file not found: ${file}" >&2
        exit 1
    fi
done

# Create log directory if it doesn't exist
LOG_DIR="${SCRIPT_DIR}/log"
mkdir -p "${LOG_DIR}"

# Get current timestamp in YYMMDDHHMM format
timestamp=$(date +%y%m%d%H%M)

# Create experiment directory
exp_dir="${EXP_DIR:-${SCRIPT_DIR}/exp/combined-ft-${dataset}-${model}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-ep${epoch}-bs${batch_size}-lda${lr_adapt}-ls${label_smooth}-mix${mixup}-${freqm}-${timem}-${timestamp}}"
mkdir -p "${exp_dir}"

# Log configuration
echo "=== Training Configuration ===" | tee "${exp_dir}/config.log"
echo "Script directory: ${SCRIPT_DIR}" | tee -a "${exp_dir}/config.log"
echo "Project root: ${PROJECT_ROOT}" | tee -a "${exp_dir}/config.log"
echo "Experiment directory: ${exp_dir}" | tee -a "${exp_dir}/config.log"
echo "Model: ${model} (${model_size})" | tee -a "${exp_dir}/config.log"
echo "Dataset: ${dataset}" | tee -a "${exp_dir}/config.log"
echo "Training data: ${tr_data}" | tee -a "${exp_dir}/config.log"
echo "Validation data: ${te_data}" | tee -a "${exp_dir}/config.log"
echo "Number of classes: ${n_class}" | tee -a "${exp_dir}/config.log"
echo "Learning rate: ${lr}" | tee -a "${exp_dir}/config.log"
echo "Batch size: ${batch_size}" | tee -a "${exp_dir}/config.log"
echo "Epochs: ${epoch}" | tee -a "${exp_dir}/config.log"
echo "=============================" | tee -a "${exp_dir}/config.log"

# Run training
echo "Starting training at $(date)" | tee -a "${exp_dir}/config.log"

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
  --weight_decay "${weight_decay}" 2>&1 | tee "${exp_dir}/training.log"

# Check if training was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Training completed successfully at $(date)" | tee -a "${exp_dir}/config.log"
    
    # Save final model info
    echo "=== Final Model Information ===" >> "${exp_dir}/config.log"
    if [ -f "${exp_dir}/best_audio_model.pth" ]; then
        echo "Best model saved at: ${exp_dir}/best_audio_model.pth" >> "${exp_dir}/config.log"
        ls -lh "${exp_dir}/best_audio_model.pth" >> "${exp_dir}/config.log"
    fi
    
    # Copy configuration for reproducibility
    cp "${BASH_SOURCE[0]}" "${exp_dir}/training_script.sh"
else
    echo "Training failed at $(date)" | tee -a "${exp_dir}/config.log"
    exit 1
fi