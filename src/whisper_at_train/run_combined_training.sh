#!/bin/bash

# SLURM 설정 (클러스터 환경에서만 사용)
# 로컬 실행 시 아래 라인들을 주석 처리하세요
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="whisper-at-combined"
#SBATCH --output=./log/%j_combined.txt

set -e  # 에러 발생 시 스크립트 중단
set -x  # 실행 명령어 출력

# 가상환경 활성화
# Poetry 환경이 있다면 자동으로 찾아서 활성화
if command -v poetry &> /dev/null && [ -f "pyproject.toml" ]; then
    echo "Poetry 환경 활성화 중..."
    poetry shell
else
    # 수동 가상환경 경로 (환경에 맞게 수정 필요)
    VENV_PATH="${HOME}/.cache/pypoetry/virtualenvs/whisper-at-z6hdRBdT-py3.10/bin/activate"
    if [ -f "$VENV_PATH" ]; then
        source "$VENV_PATH"
    else
        echo "경고: 가상환경을 찾을 수 없습니다. 시스템 Python을 사용합니다."
    fi
fi
# 환경 설정
export TORCH_HOME=${TORCH_HOME:-"../../pretrained_models"}

# 하이퍼파라미터 설정
lr=1e-6
freqm=0
timem=10
mixup=0.5
batch_size=48
model=whisper-high-lw_tr_1_8  # TL-TR 아키텍처 모델
model_size=large-v1

# 데이터셋 설정
dataset=audioset_sonyc
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

# 적응형 학습률 설정
lr_adapt=True
lr_patience=2

# 데이터 경로 설정 - 환경에 맞게 수정 가능
DATA_ROOT="${DATA_ROOT:-/mnt/ssd_disk/github/whisper-at/data/processed_data}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/ssd_disk/github/whisper-at/pretrained_models}"

tr_data="${DATA_ROOT}/combined_train.json"
te_data="${DATA_ROOT}/combined_val.json"
label_csv="${DATA_ROOT}/class_labels_indices_extended.csv"
pretrained_model="${MODEL_ROOT}/large-v1_ori.pth"

# 클래스 설정
n_class=533  # AudioSet (527) + SONYC-UST (6) 확장 클래스
label_smooth=0.1

# 파일 존재 확인
echo "데이터 파일 존재 확인 중..."
for file in "$tr_data" "$te_data" "$label_csv" "$pretrained_model"; do
    if [ ! -f "$file" ]; then
        echo "오류: 파일을 찾을 수 없습니다: $file"
        echo "DATA_ROOT 또는 MODEL_ROOT 환경변수를 확인하세요."
        exit 1
    fi
done

# 실험 디렉토리 설정
timestamp=$(date +%y%m%d%H%M)
exp_dir=./exp/combined-ft-${dataset}-${model}-${model_size}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-ep${epoch}-bs${batch_size}-lda${lr_adapt}-ls${label_smooth}-mix${mixup}-${freqm}-${timem}-${timestamp}

echo "실험 디렉토리 생성: $exp_dir"
mkdir -p "$exp_dir"
mkdir -p "./log"

# 실험 설정을 파일로 저장
cat > "$exp_dir/config.txt" << EOF
실험 설정:
- 모델: $model ($model_size)
- 데이터셋: $dataset
- 학습률: $lr
- 에포크: $epoch
- 배치 크기: $batch_size
- 클래스 수: $n_class
- 타임스탬프: $timestamp
EOF

# 학습 시작
echo "=== Whisper-AT 결합 학습 시작 ==="
echo "학습 데이터: $tr_data"
echo "검증 데이터: $te_data"
echo "모델: $model ($model_size)"
echo "실험 디렉토리: $exp_dir"
echo "=================================="

python -W ignore ./run.py \
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
    --weight_decay "${weight_decay}"

# 학습 완료 후 결과 확인
if [ $? -eq 0 ]; then
    echo "학습이 성공적으로 완료되었습니다!"
    echo "결과는 다음 디렉토리에 저장되었습니다: $exp_dir"
    ls -la "$exp_dir"
else
    echo "학습 중 오류가 발생했습니다."
    exit 1
fi