#!/bin/bash
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="w-as-high"
#SBATCH --output=./log/%j_as.txt
#SBATCH --error=./log/%j_as_err.txt
#SBATCH --time=48:00:00

set -x
set -e  # Exit on any error

# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc

# Activate virtual environment
if [ -f "/home/taemyung_heo/.cache/pypoetry/virtualenvs/whisper-at-z6hdRBdT-py3.10/bin/activate" ]; then
    source /home/taemyung_heo/.cache/pypoetry/virtualenvs/whisper-at-z6hdRBdT-py3.10/bin/activate
else
    echo "Virtual environment not found. Please check the path."
    exit 1
fi

export TORCH_HOME=../../pretrained_models
export CUDA_VISIBLE_DEVICES=0

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

# Data paths - use environment variable or default
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

# Verify critical files exist
for file in "$tr_data" "$te_data" "$label_csv" "$pretrained_model"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file not found: $file"
        exit 1
    fi
done

# Get current timestamp in YYMMDDHHMM format
timestamp=$(date +%y%m%d%H%M)

exp_dir=./exp/combined-ft-${dataset}-${model}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-ep${epoch}-bs${batch_size}-lda${lr_adapt}-ls${label_smooth}-mix${mixup}-${freqm}-${timem}-${timestamp}
mkdir -p $exp_dir
mkdir -p ./log

# Copy this script to exp_dir for reproducibility
cp "$0" "${exp_dir}/run_script.sh"

# Log experiment parameters
echo "Experiment started at: $(date)" > "${exp_dir}/experiment_info.txt"
echo "Model: ${model}" >> "${exp_dir}/experiment_info.txt"
echo "Model size: ${model_size}" >> "${exp_dir}/experiment_info.txt"
echo "Learning rate: ${lr}" >> "${exp_dir}/experiment_info.txt"
echo "Batch size: ${batch_size}" >> "${exp_dir}/experiment_info.txt"
echo "Epochs: ${epoch}" >> "${exp_dir}/experiment_info.txt"

echo "Starting training with experiment directory: $exp_dir"
python -W ignore -u ./run.py \
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