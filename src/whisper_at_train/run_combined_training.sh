#!/bin/bash
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="w-as-high"
#SBATCH --output=./log/%j_as.txt

set -e  # Exit on any error
set -x  # Print commands
set -o pipefail  # Detect errors in pipelines

# Get the script directory for relative path resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Change to script directory to ensure relative paths work correctly
cd "$SCRIPT_DIR"

# Environment setup - check if running in Poetry environment
if command -v poetry &> /dev/null && poetry env info --path &> /dev/null; then
    echo "Using Poetry environment"
    # Use source instead of poetry shell for non-interactive environments
    # shellcheck source=/dev/null
    source "$(poetry env info --path)/bin/activate"
else
    # Fallback: require user-provided VENV_PATH
    if [[ -z "${VENV_PATH}" ]]; then
        echo "Error: Poetry env not found and VENV_PATH is not set."
        echo "Please set VENV_PATH environment variable to your virtual environment activate script."
        exit 1
    fi
    if [[ -f "$VENV_PATH" ]]; then
        # shellcheck source=/dev/null
        source "$VENV_PATH"
    else
        echo "Error: Virtual environment not found at $VENV_PATH"
        exit 1
    fi
fi

export TORCH_HOME="${PROJECT_ROOT}/pretrained_models"

lr=1e-6
freqm=0
timem=10
mixup=0.5
batch_size=48
model=whisper-high-lw_tr_1_8 #whisper-high-lw_tr_1_8 (tl-tr, lr=5e-5) whisper-high-lw_down_tr_512_1_8 (tl-tr-512, w/ low-dim proj, lr=1e-4)
model_size=large-v1

dataset=audioset_sonyc
bal=none
epoch=50
weight_decay=1e-5
lrscheduler_start=15
lrscheduler_decay=0.75
lrscheduler_step=5
wa=True
wa_start=36
wa_end=50
lr_adapt=True
lr_patience=2

# Data paths - use environment variables with fallbacks
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/processed_data}"
tr_data="${TRAIN_DATA:-${DATA_DIR}/combined_train.json}"
te_data="${VAL_DATA:-${DATA_DIR}/combined_val.json}"
label_csv="${LABEL_CSV:-${DATA_DIR}/class_labels_indices_extended.csv}"
n_class=533
label_smooth=0.1

# Model paths
PRETRAINED_DIR="${PRETRAINED_DIR:-${PROJECT_ROOT}/pretrained_models}"
pretrained_model="${PRETRAINED_MODEL:-${PRETRAINED_DIR}/large-v1_ori.pth}"

# Validate required files exist
for file in "$tr_data" "$te_data" "$label_csv" "$pretrained_model"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: Required file not found: $file"
        exit 1
    fi
done

# Get current timestamp in YYMMDDHHMM format
timestamp=$(date +%y%m%d%H%M)

# Create experiment directory
exp_dir="${EXP_DIR:-${PROJECT_ROOT}/exp}/combined-ft-${dataset}-${model}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-ep${epoch}-bs${batch_size}-lda${lr_adapt}-ls${label_smooth}-mix${mixup}-${freqm}-${timem}-${timestamp}"
if ! mkdir -p "$exp_dir"; then
    echo "Error: Cannot create experiment directory $exp_dir"
    exit 1
fi

echo "Starting training with experiment directory: $exp_dir"
echo "Training data: $tr_data"
echo "Validation data: $te_data"
echo "Pretrained model: $pretrained_model"

# Run training with error handling
if ! python -W ignore ./run.py \
  --model "${model}" \
  --dataset "${dataset}" \
  --data-train "${tr_data}" \
  --data-val "${te_data}" \
  --exp-dir "$exp_dir" \
  --label-csv "${label_csv}" \
  --n_class ${n_class} \
  --lr ${lr} \
  --n-epochs ${epoch} \
  --batch-size ${batch_size} \
  --save_model True \
  --freqm ${freqm} \
  --timem ${timem} \
  --mixup ${mixup} \
  --bal "${bal}" \
  --model_size "${model_size}" \
  --label_smooth ${label_smooth} \
  --lrscheduler_start ${lrscheduler_start} \
  --lrscheduler_decay ${lrscheduler_decay} \
  --lrscheduler_step ${lrscheduler_step} \
  --loss BCE \
  --metrics mAP \
  --warmup True \
  --wa ${wa} \
  --wa_start ${wa_start} \
  --wa_end ${wa_end} \
  --lr_adapt ${lr_adapt} \
  --lr_patience ${lr_patience} \
  --num-workers 8 \
  --pretrained_model "${pretrained_model}" \
  --weight_decay ${weight_decay}; then
    echo "Training failed! Check logs in $exp_dir"
    exit 1
fi

echo "Training completed successfully! Results saved in: $exp_dir"