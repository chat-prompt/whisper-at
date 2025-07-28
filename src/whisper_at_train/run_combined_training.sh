#!/bin/bash
# Combined training script for Whisper-AT models
# Trains on AudioSet + SONYC-UST combined dataset with TL-TR architecture
#
# SLURM configuration:
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="w-as-high"
#SBATCH --output=./log/%j_as.txt

set -e  # Exit on any error
set -x  # Print commands

# Activate poetry environment (modify path as needed)
VENV_PATH="/home/taemyung_heo/.cache/pypoetry/virtualenvs/whisper-at-z6hdRBdT-py3.10/bin/activate"
if [[ -f "$VENV_PATH" ]]; then
    source "$VENV_PATH"
    echo "Poetry environment activated"
else
    echo "Poetry environment not found at $VENV_PATH, using system Python"
fi

# Set TORCH_HOME relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TORCH_HOME="$SCRIPT_DIR/../../pretrained_models"
echo "TORCH_HOME set to: $TORCH_HOME"

# Training hyperparameters
lr=1e-6                    # Learning rate
freqm=0                    # Frequency masking
timem=10                   # Time masking  
mixup=0.5                  # Mixup augmentation
batch_size=48              # Training batch size
model=whisper-high-lw_tr_1_8  # Model architecture (TL-TR)
model_size=large-v1        # Whisper base model size

# Dataset and training configuration
dataset=audioset_sonyc     # Combined AudioSet + SONYC-UST dataset
bal=none                   # Class balancing strategy
epoch=50                   # Number of training epochs
weight_decay=1e-5          # L2 regularization
lrscheduler_start=15       # When to start learning rate decay
lrscheduler_decay=0.75     # Learning rate decay factor
lrscheduler_step=5         # Learning rate decay step
wa=True                    # Weight averaging
wa_start=36                # Weight averaging start epoch
wa_end=50                  # Weight averaging end epoch
lr_adapt=True              # Adaptive learning rate
lr_patience=2              # Learning rate adaptation patience
# Data paths - convert to relative paths from script directory
DATA_DIR="$SCRIPT_DIR/../../data/processed_data"
PRETRAINED_DIR="$SCRIPT_DIR/../../pretrained_models"

tr_data="$DATA_DIR/combined_train.json"
te_data="$DATA_DIR/combined_val.json"
label_csv="$DATA_DIR/class_labels_indices_extended.csv"
pretrained_model="$PRETRAINED_DIR/large-v1_ori.pth"

# Verify data files exist
for file in "$tr_data" "$te_data" "$label_csv" "$pretrained_model"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: Required file not found: $file"
        exit 1
    fi
done
echo "All required data files verified"

n_class=533
label_smooth=0.1

# Get current timestamp in YYMMDDHHMM format
timestamp=$(date +%y%m%d%H%M)

exp_dir=./exp/combined-ft-${dataset}-${model}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-ep${epoch}-bs${batch_size}-lda${lr_adapt}-ls${label_smooth}-mix${mixup}-${freqm}-${timem}-${timestamp}
mkdir -p $exp_dir

echo "Starting training with the following configuration:"
echo "Model: $model ($model_size)"
echo "Dataset: $dataset"
echo "Learning rate: $lr"
echo "Epochs: $epoch"
echo "Batch size: $batch_size"
echo "Experiment directory: $exp_dir"
echo "Training data: $tr_data"
echo "Validation data: $te_data"
echo

python -W ignore ./run.py \
  --model ${model} \
  --dataset ${dataset} \
  --data-train ${tr_data} \
  --data-val ${te_data} \
  --exp-dir $exp_dir \
  --label-csv ${label_csv} \
  --n_class ${n_class} \
  --lr $lr \
  --n-epochs ${epoch} \
  --batch-size ${batch_size} \
  --save_model True \
  --freqm ${freqm} \
  --timem ${timem} \
  --mixup ${mixup} \
  --bal ${bal} \
  --model_size ${model_size} \
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
  --pretrained_model ${pretrained_model} \
  --weight_decay ${weight_decay}
