#!/bin/bash
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="whisper-at-combined"
#SBATCH --output=./log/%j_combined.txt

set -e # Exit on any error
set -x # Print commands

# Environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate Poetry environment (adjust path as needed)
if [ -f "/home/taemyung_heo/.cache/pypoetry/virtualenvs/whisper-at-z6hdRBdT-py3.10/bin/activate" ]; then
    source /home/taemyung_heo/.cache/pypoetry/virtualenvs/whisper-at-z6hdRBdT-py3.10/bin/activate
else
    echo "Poetry virtual environment not found. Using system Python."
fi

export TORCH_HOME="$PROJECT_ROOT/pretrained_models"

# Training hyperparameters
lr=1e-6
freqm=0
timem=10
mixup=0.5
batch_size=48
weight_decay=1e-5
label_smooth=0.1

# Model configuration
model=whisper-high-lw_tr_1_8  # Time and Layer-wise Transformer
model_size=large-v1

# Dataset configuration
dataset=audioset_sonyc
bal=none
n_class=533  # 527 AudioSet + 6 SONYC-UST classes

# Training schedule
epoch=50
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

# Data paths (use relative paths based on PROJECT_ROOT)
DATA_DIR="$PROJECT_ROOT/data/processed_data"
tr_data="$DATA_DIR/combined_train.json"
te_data="$DATA_DIR/combined_val.json"
label_csv="$DATA_DIR/class_labels_indices_extended.csv"
pretrained_model="$PROJECT_ROOT/pretrained_models/large-v1_ori.pth"

# Validate required files and directories
if [ ! -f "$tr_data" ]; then
    echo "Error: Training data file not found: $tr_data"
    exit 1
fi

if [ ! -f "$te_data" ]; then
    echo "Error: Validation data file not found: $te_data"
    exit 1
fi

if [ ! -f "$label_csv" ]; then
    echo "Error: Label CSV file not found: $label_csv"
    exit 1
fi

if [ ! -f "$pretrained_model" ]; then
    echo "Error: Pretrained model file not found: $pretrained_model"
    exit 1
fi

# Create experiment directory
timestamp=$(date +%y%m%d%H%M)
exp_dir="$SCRIPT_DIR/exp/combined-ft-${dataset}-${model}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-ep${epoch}-bs${batch_size}-lda${lr_adapt}-ls${label_smooth}-mix${mixup}-${freqm}-${timem}-${timestamp}"
mkdir -p "$exp_dir"
mkdir -p "$SCRIPT_DIR/log"

echo "Starting training with experiment directory: $exp_dir"

# Run training
python -W ignore ./run.py \
  --model "${model}" \
  --dataset "${dataset}" \
  --data-train "${tr_data}" \
  --data-val "${te_data}" \
  --exp-dir "${exp_dir}" \
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

# Check training completion status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved in: $exp_dir"
else
    echo "Training failed with exit code $?"
    exit 1
fi