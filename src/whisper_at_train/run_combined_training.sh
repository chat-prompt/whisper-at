#!/bin/bash
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="w-as-high"
#SBATCH --output=./log/%j_as.txt

set -euo pipefail

# Poetry 가상환경 활성화 (동적 감지)
if command -v poetry >/dev/null 2>&1; then
    echo "Poetry 가상환경 활성화 중..."
    eval "$(poetry env info --path)/bin/activate" 2>/dev/null || {
        echo "Poetry 환경을 찾을 수 없습니다. poetry shell을 먼저 실행하세요."
        exit 1
    }
fi

# 프로젝트 루트 디렉토리 기준으로 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export TORCH_HOME="$PROJECT_ROOT/pretrained_models"

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
# 데이터 경로 설정 (프로젝트 루트 기준)
tr_data="$PROJECT_ROOT/data/processed_data/combined_train.json"
te_data="$PROJECT_ROOT/data/processed_data/combined_val.json"
label_csv="$PROJECT_ROOT/data/processed_data/class_labels_indices_extended.csv"
n_class=533
label_smooth=0.1

pretrained_model="$PROJECT_ROOT/pretrained_models/large-v1_ori.pth"

# 필수 파일들 존재 확인
for file in "$tr_data" "$te_data" "$label_csv" "$pretrained_model"; do
    if [[ ! -f "$file" ]]; then
        echo "오류: 필수 파일을 찾을 수 없습니다: $file"
        exit 1
    fi
done

# 실험 디렉토리 설정
timestamp=$(date +%y%m%d%H%M)
exp_dir="./exp/combined-ft-${dataset}-${model}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-ep${epoch}-bs${batch_size}-lda${lr_adapt}-ls${label_smooth}-mix${mixup}-${freqm}-${timem}-${timestamp}"

echo "실험 디렉토리 생성: $exp_dir"
mkdir -p "$exp_dir" || {
    echo "오류: 실험 디렉토리 생성 실패: $exp_dir"
    exit 1
}

# 로그 디렉토리 생성
mkdir -p "./log"

# 훈련 실행
echo "=========================================="
echo "Whisper-AT 결합 훈련 시작"
echo "모델: ${model} (${model_size})"
echo "데이터셋: ${dataset}"
echo "배치 크기: ${batch_size}, 에포크: ${epoch}"
echo "학습률: ${lr}, 가중치 감쇠: ${weight_decay}"
echo "실험 디렉토리: $exp_dir"
echo "=========================================="

python -W ignore ./run.py \
  --model "${model}" \
  --dataset "${dataset}" \
  --data-train "${tr_data}" \
  --data-val "${te_data}" \
  --exp-dir "$exp_dir" \
  --label-csv "${label_csv}" \
  --n_class "${n_class}" \
  --lr "$lr" \
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

# 훈련 완료 상태 확인
if [[ $? -eq 0 ]]; then
    echo "=========================================="
    echo "훈련이 성공적으로 완료되었습니다!"
    echo "결과는 다음 디렉토리에 저장됩니다: $exp_dir"
    echo "=========================================="
else
    echo "=========================================="
    echo "오류: 훈련 중 문제가 발생했습니다."
    echo "로그를 확인하세요: ./log/"
    echo "=========================================="
    exit 1
fi