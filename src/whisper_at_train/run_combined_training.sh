#!/bin/bash
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="w-as-combined"
#SBATCH --output=./log/%j_combined.txt
#SBATCH --error=./log/%j_combined.err
#SBATCH --time=48:00:00

set -x
set -e  # Exit on error

# Python 환경 설정
if [ -f "/home/taemyung_heo/.cache/pypoetry/virtualenvs/whisper-at-z6hdRBdT-py3.10/bin/activate" ]; then
    source /home/taemyung_heo/.cache/pypoetry/virtualenvs/whisper-at-z6hdRBdT-py3.10/bin/activate
else
    echo "Error: Virtual environment not found" >&2
    exit 1
fi

# PyTorch 모델 캐시 디렉토리 설정
export TORCH_HOME="${TORCH_HOME:-../../pretrained_models}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 스크립트 디렉토리를 기준으로 한 상대 경로 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 하이퍼파라미터 설정
lr=1e-6
freqm=0
timem=10
mixup=0.5
batch_size=48
model=whisper-high-lw_tr_1_8  # 사용 가능 모델: whisper-high-lw_tr_1_8 (tl-tr), whisper-high-lw_down_tr_512_1_8 (tl-tr-512)
model_size=large-v1

# 데이터셋 설정
dataset=audioset_sonyc  # AudioSet + SONYC-UST 결합 데이터셋
bal=none
epoch=50
weight_decay=1e-5

# 학습률 스케줄러 설정
lrscheduler_start=15
lrscheduler_decay=0.75
lrscheduler_step=5

# Weight Averaging 설정
wa=True
wa_start=36
wa_end=50

# Adaptive Learning Rate 설정
lr_adapt=True
lr_patience=2

# 데이터 경로 설정 (프로젝트 루트 기준)
tr_data="${PROJECT_ROOT}/data/processed_data/combined_train.json"
te_data="${PROJECT_ROOT}/data/processed_data/combined_val.json"
label_csv="${PROJECT_ROOT}/data/processed_data/class_labels_indices_extended.csv"
n_class=533  # AudioSet(527) + SONYC-UST(6) = 533 클래스
label_smooth=0.1

# 사전 학습 모델 경로
pretrained_model="${PROJECT_ROOT}/pretrained_models/large-v1_ori.pth"

# 필수 파일 및 디렉토리 존재 확인
check_file() {
    if [ ! -f "$1" ]; then
        echo "Error: Required file not found: $1" >&2
        exit 1
    fi
}

check_dir() {
    if [ ! -d "$1" ]; then
        echo "Error: Required directory not found: $1" >&2
        exit 1
    fi
}

# 데이터 파일 검증
check_file "$tr_data"
check_file "$te_data"
check_file "$label_csv"
check_file "$pretrained_model"

# 타임스탬프 생성
timestamp=$(date +%y%m%d%H%M)

# 실험 디렉토리 설정
exp_dir="${SCRIPT_DIR}/exp/combined-ft-${dataset}-${model}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-ep${epoch}-bs${batch_size}-lda${lr_adapt}-ls${label_smooth}-mix${mixup}-${freqm}-${timem}-${timestamp}"
mkdir -p "$exp_dir"
mkdir -p "${SCRIPT_DIR}/log"  # 로그 디렉토리 생성

# 실험 설정 로깅
echo "=== Experiment Configuration ===" | tee "${exp_dir}/config.log"
echo "Timestamp: $(date)" | tee -a "${exp_dir}/config.log"
echo "Model: ${model} (${model_size})" | tee -a "${exp_dir}/config.log"
echo "Dataset: ${dataset} (${n_class} classes)" | tee -a "${exp_dir}/config.log"
echo "Learning Rate: ${lr}" | tee -a "${exp_dir}/config.log"
echo "Batch Size: ${batch_size}" | tee -a "${exp_dir}/config.log"
echo "Epochs: ${epoch}" | tee -a "${exp_dir}/config.log"
echo "Experiment Directory: ${exp_dir}" | tee -a "${exp_dir}/config.log"
echo "================================" | tee -a "${exp_dir}/config.log"

# Python 스크립트 실행
python -W ignore "${SCRIPT_DIR}/run.py" \
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

# 실행 결과 확인
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "Training completed successfully!" | tee -a "${exp_dir}/config.log"
    echo "Results saved to: ${exp_dir}" | tee -a "${exp_dir}/config.log"
else
    echo "Training failed with exit code: $exit_code" | tee -a "${exp_dir}/config.log"
    exit $exit_code
fi