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
source /home/taemyung_heo/.cache/pypoetry/virtualenvs/whisper-at-z6hdRBdT-py3.10/bin/activate
export TORCH_HOME=../../pretrained_models

lr=1e-6
freqm=48
timem=120
mixup=0.3
batch_size=32
model=whisper-high-lw_tr_1_8 #whisper-high-lw_tr_1_8 (tl-tr, lr=5e-5) whisper-high-lw_down_tr_512_1_8 (tl-tr-512, w/ low-dim proj, lr=1e-4)
model_size=large-v1

dataset=sonyc
bal=none
epoch=30
weight_decay=1e-4
lrscheduler_start=15
lrscheduler_decay=0.8
lrscheduler_step=5
wa=True
wa_start=15
wa_end=30
lr_adapt=False
tr_data=/mnt/ssd_disk/github/whisper-at/data/processed_data/sonyc_train.json
te_data=/mnt/ssd_disk/github/whisper-at/data/processed_data/combined_val.json
label_csv=/mnt/ssd_disk/github/whisper-at/data/processed_data/class_labels_indices_extended.csv
n_class=533
label_smooth=0.1

pretrained_model=/mnt/ssd_disk/github/whisper-at/pretrained_models/large-v1_ori.pth

# Get current timestamp in YYMMDDHHMM format
timestamp=$(date +%y%m%d%H%M)

exp_dir=./exp/sonyc-ft-${dataset}-${model}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-ep${epoch}-bs${batch_size}-lda${lr_adapt}-ls${label_smooth}-mix${mixup}-${freqm}-${timem}-${timestamp}
mkdir -p $exp_dir

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
  --num-workers 8 \
  --pretrained_model ${pretrained_model} \
  --weight_decay ${weight_decay}
  #--freeze_original_classes