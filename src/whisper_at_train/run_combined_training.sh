#!/bin/bash
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="whisper-at-combined"
#SBATCH --output=./log/%j_combined.txt

set -e  # Exit on any error
set -x  # Print commands for debugging

# Activate poetry environment if available, otherwise use system python
if command -v poetry &> /dev/null; then
    echo "Using poetry environment..."
    source $(poetry env info --path)/bin/activate
else
    echo "Poetry not found, using system python"
fi

# Set TORCH_HOME relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TORCH_HOME="${SCRIPT_DIR}/../../pretrained_models"

lr=1e-6
freqm=0
timem=10
mixup=0.5
batch_size=48
# Model configuration
model=whisper-high-lw_tr_1_8
model_size=large-v1

# Training configuration
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
n_class=533
label_smooth=0.1

# Data paths - make them relative to script location for portability
DATA_BASE_DIR="${SCRIPT_DIR}/../../data/processed_data"
PRETRAINED_BASE_DIR="${SCRIPT_DIR}/../../pretrained_models"

tr_data="${DATA_BASE_DIR}/combined_train.json"
te_data="${DATA_BASE_DIR}/combined_val.json"
label_csv="${DATA_BASE_DIR}/class_labels_indices_extended.csv"
pretrained_model="${PRETRAINED_BASE_DIR}/large-v1_ori.pth"

# Validate required files exist
echo "Validating required files..."
for file in "$tr_data" "$te_data" "$label_csv" "$pretrained_model"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: Required file not found: $file"
        exit 1
    fi
done
echo "All required files found."

# Create experiment directory with timestamp
timestamp=$(date +%y%m%d%H%M)
exp_dir="./exp/combined-ft-${dataset}-${model}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-ep${epoch}-bs${batch_size}-lda${lr_adapt}-ls${label_smooth}-mix${mixup}-${freqm}-${timem}-${timestamp}"

echo "Creating experiment directory: $exp_dir"
mkdir -p "$exp_dir"

# Log configuration
echo "Starting training with configuration:"
echo "Model: $model ($model_size)"
echo "Dataset: $dataset"
echo "Epochs: $epoch, Batch size: $batch_size"
echo "Learning rate: $lr"
echo "Experiment directory: $exp_dir"

# Start training
echo "Starting training..."
python -W ignore ./run.py \
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
  --weight_decay ${weight_decay}

echo "Training completed. Results saved to: $exp_dir"