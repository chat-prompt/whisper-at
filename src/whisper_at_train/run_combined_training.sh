#!/bin/bash
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="w-as-high"
#SBATCH --output=./log/%j_as.txt

set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
# Use poetry environment if available, otherwise use conda/venv
if command -v poetry &> /dev/null; then
    poetry shell
elif [ -f "/home/taemyung_heo/.cache/pypoetry/virtualenvs/whisper-at-z6hdRBdT-py3.10/bin/activate" ]; then
    source /home/taemyung_heo/.cache/pypoetry/virtualenvs/whisper-at-z6hdRBdT-py3.10/bin/activate
else
    echo "Warning: Poetry environment not found. Please activate your virtual environment manually."
fi
export TORCH_HOME=../../pretrained_models

# Learning rate based on model type
if [[ "$model" == *"down_tr"* ]]; then
    lr=1e-4  # Higher LR for low-dim projection models
else
    lr=5e-5  # Standard LR for TL-TR models
fi
freqm=0
timem=10
mixup=0.5
batch_size=48
# Model selection: whisper-high-lw_tr_1_8 (tl-tr) or whisper-high-lw_down_tr_512_1_8 (tl-tr-512)
model=whisper-high-lw_tr_1_8
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
# Dataset paths - use environment variable if set, otherwise use default
DATA_ROOT=${DATA_ROOT:-/mnt/ssd_disk/github/whisper-at}
tr_data=${DATA_ROOT}/data/processed_data/combined_train.json
#tr_data=${DATA_ROOT}/data/processed_data/audioset_train.json
te_data=${DATA_ROOT}/data/processed_data/combined_val.json
label_csv=${DATA_ROOT}/data/processed_data/class_labels_indices_extended.csv
#label_csv=${DATA_ROOT}/audioset_label.csv
n_class=533
# n_class=527
label_smooth=0.1

pretrained_model=${DATA_ROOT}/pretrained_models/large-v1_ori.pth

# Validate required files exist
if [ ! -f "$tr_data" ]; then
    echo "Error: Training data not found at $tr_data"
    exit 1
fi
if [ ! -f "$te_data" ]; then
    echo "Error: Validation data not found at $te_data"
    exit 1
fi
if [ ! -f "$label_csv" ]; then
    echo "Error: Label CSV not found at $label_csv"
    exit 1
fi
if [ ! -f "$pretrained_model" ]; then
    echo "Error: Pretrained model not found at $pretrained_model"
    exit 1
fi

# Get current timestamp in YYMMDDHHMM format
timestamp=$(date +%y%m%d%H%M)

exp_dir=./exp/combined-ft-${dataset}-${model}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-ep${epoch}-bs${batch_size}-lda${lr_adapt}-ls${label_smooth}-mix${mixup}-${freqm}-${timem}-${timestamp}
mkdir -p $exp_dir

# Log experiment configuration
echo "Starting training with configuration:" | tee ${exp_dir}/config.log
echo "Model: ${model}" | tee -a ${exp_dir}/config.log
echo "Model size: ${model_size}" | tee -a ${exp_dir}/config.log
echo "Learning rate: ${lr}" | tee -a ${exp_dir}/config.log
echo "Batch size: ${batch_size}" | tee -a ${exp_dir}/config.log
echo "Epochs: ${epoch}" | tee -a ${exp_dir}/config.log
echo "Dataset: ${dataset}" | tee -a ${exp_dir}/config.log
echo "Timestamp: ${timestamp}" | tee -a ${exp_dir}/config.log

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
  #--freeze_original_classes

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved to: ${exp_dir}"
else
    echo "Training failed with exit code: $?"
    exit 1
fi