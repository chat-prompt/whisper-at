# Whisper-AT Extended: 종합 음향 인식 기술 개발

> **원본 프로젝트**: [YuanGongND/whisper-at](https://github.com/YuanGongND/whisper-at)
> **본 저장소**: [chat-prompt/whisper-at](https://github.com/chat-prompt/whisper-at)

## 프로젝트 개요

이 프로젝트는 [Whisper-AT](https://github.com/YuanGongND/whisper-at) 논문 구현체를 fork하여 **TIPS 종합 음향 인식 기술 개발 연구과제**를 수행한 결과물입니다.

### 연구 목표
- OpenAI Whisper 기반 음성 인식 성능을 유지하면서 음향 이벤트 태깅 기능 강화
- 도시 소음 환경(SONYC-UST)에 대한 음향 인식 성능 향상
- 웨비나/강의 환경에서 5가지 핵심 음향(박수, 웃음, 기침, 한숨, 환호) 100% 정확도 달성
- 음성 인식 성능 저하 없이 음향 태깅 기능 통합

### 연구 필요성
- 온라인 콘텐츠 자동 분석 및 리퍼포징 수요 증가
- 음성뿐만 아니라 음향 정보를 활용한 멀티모달 분석 필요
- 웨비나, 온라인 강의 등에서 청중 반응 자동 분석 요구

---

## 연구개발 전체 과정

### 1단계: 기반 연구 및 기술 분석

**수행 내용:**
- OpenAI Whisper 및 Whisper-AT 핵심 논문 심층 분석
- 음성 인식과 음향 태깅 동시 수행 가능성 검증
- 공식 GitHub 저장소(YuanGongND/whisper-at) 코드 분석
- 기술적 타당성 검토 및 연구 방향 수립

**주요 발견:**
- Whisper는 노이즈에 강인하지만 noise-invariant가 아닌 **noise-variant** 표현을 학습
- 원본 Whisper 파라미터를 동결하고 TL-TR(Time and Layer-wise Transformer)만 학습하여 계산 비용 최소화
- AudioSet 527개 클래스에 대해 mAP 0.404~0.421 성능 달성 가능

**관련 자료:**
- `report/2212.04356v1.pdf` - Whisper 원본 논문
- `report/2307.03183v1.pdf` - Whisper-AT 원본 논문
- `report/Whisper_논문_리뷰.pdf` - Whisper 논문 상세 리뷰
- `report/Whisper-AT_논문_리뷰.pdf` - Whisper-AT 논문 상세 리뷰

### 2단계: 자체 데이터셋 구축 시도 및 전략 전환

**수행 내용:**
- 웨비나/온라인 스터디 환경 특화 음향 데이터 수집
- 데이터 레이블링 프로세스 설계 및 품질 관리 시도
- 데이터셋 구축의 현실적 한계 분석

**직면한 과제:**
- 데이터 다양성 및 품질 확보의 어려움
- 레이블링 복잡성 및 시간 소요
- 데이터 편향성 문제
- 대규모 고품질 데이터셋 구축의 현실적 제약

**전략적 결정:**
- SONYC-UST 공개 데이터셋 활용으로 방향 전환
- 검증된 데이터셋을 통한 연구 효율성 확보

### 3단계: SONYC-UST 데이터셋 통합 및 모델 확장

**수행 내용:**
- SONYC-UST (Sounds of New York City - Urban Sound Tagging) 데이터셋 분석
- 6개 신규 도시 소음 클래스 정의 및 레이블 체계 구축
- AudioSet 527개 + SONYC-UST 6개 = **총 533개 클래스로 확장**
- 클래스 레이블 매핑 파일 생성 및 데이터 전처리

**신규 SONYC-UST 6개 클래스:**

| Index | 클래스명 | 설명 |
|-------|---------|------|
| 527 | Amplified speech | 증폭된 음성 |
| 528 | Hoe ram | 굴착기 파쇄기 |
| 529 | Large rotating saw | 대형 회전톱 |
| 530 | Non machinery impact | 비기계적 충격음 |
| 531 | Pile driver | 항타기 |
| 532 | Small medium rotating saw | 중소형 회전톱 |

**관련 데이터 파일:**
- `data/processed_data/audioset_train.json` - AudioSet 20k 학습 데이터
- `data/processed_data/sonyc_new_train.json` - SONYC-UST 학습 데이터
- `data/processed_data/combined_train.json` - AudioSet + SONYC-UST 결합 학습 데이터
- `data/processed_data/combined_val.json` - 결합 검증 데이터
- `data/processed_data/class_labels_indices_extended.csv` - 533개 클래스 정의 파일
- `data/processed_data/sonyc_new_class_mapping.json` - SONYC 6개 신규 클래스 매핑

### 4단계: 모델 파인튜닝 및 학습

**수행 내용:**
- Whisper-AT Large-v1 기반 TL-TR 모델 파인튜닝
- AudioSet 20k + SONYC-UST 결합 데이터셋으로 학습
- 가중치 평균화(Weight Averaging) 기법 적용 (epoch 36-50)
- 하이퍼파라미터 튜닝 및 최적화

**학습 환경:**
- OS: Debian GNU/Linux 11 (Kernel 5.10.0-35-cloud-amd64)
- GPU: NVIDIA Tesla T4 (15GB VRAM)
- CUDA 12.4, PyTorch 2.3.1
- Python 3.10.17

**주요 하이퍼파라미터:**
```
Model: whisper-high-lw_tr_1_8 (Whisper Large-v1 + TL-TR)
Learning Rate: 1e-6 (초기), ReduceLROnPlateau (patience=2, decay=0.75)
Batch Size: 48
Epochs: 50
Weight Decay: 1e-5
Label Smoothing: 0.1
Mixup: 0.5
Time Masking: 10
Loss Function: BCEWithLogitsLoss
Weight Averaging: epoch 36-50
학습 가능 파라미터: 약 4,000만 개
```

**관련 코드:**
- `src/whisper_at_train/run_combined_training.sh` - 결합 학습 실행 스크립트
- `src/whisper_at_train/run.py` - 학습 메인 진입점
- `src/whisper_at_train/traintest.py` - 학습/테스트 핵심 로직
- `src/whisper_at_train/models.py` - TL-TR 모델 아키텍처 정의
- `src/whisper_at_train/dataloader_feat.py` - 특징 기반 데이터로더

### 5단계: 모델 평가 및 성능 검증

**수행 내용:**
- 일반 음향 태깅 성능 평가 (AudioSet 527개 클래스)
- SONYC-UST 특정 클래스 성능 비교 분석
- 원본 Whisper-AT 모델 대비 성능 개선 측정

**관련 코드:**
- `src/whisper_at_train/evaluate_pretrained_whisper_at.py` - 사전학습 모델 평가 스크립트

### 6단계: TC1 테스트셋 구축 및 검증

**수행 내용:**
- 웨비나/강의 환경 특화 테스트 데이터셋 구축
- YouTube 웨비나/강의 영상 10개 선정 및 레이블링
- 5개 목표 음향 정확도 평가 프로토콜 수립

**TC1 테스트 규격:**
- **목표**: 30dB SNR 웨비나/강의 영상에서 5가지 음향(박수, 웃음, 기침, 한숨, 환호) 100% 정확도 검출
- **평가 기준**: 음향 인식 정확도 = (TP + TN) / (TP + TN + FP + FN) × 100
- **허용 오차**: 실제 음향 발생 시점 ±1초 이내 태그하면 "검출"로 판정
- **추가 조건**: 음성 인식 성능 저하 없음 (ΔWER ≤ 0.5%p)

**테스트 데이터셋 구성 (총 164개 샘플):**
| 음향 종류 | 샘플 수 | 파일 수 |
|---------|--------|--------|
| 박수 소리 | 45개 | 9개 |
| 웃음 소리 | 107개 | 5개 |
| 기침 소리 | 6개 | 1개 |
| 한숨 소리 | 3개 | 1개 |
| 환호하는 소리 | 3개 | 1개 |

**관련 문서:**
- `report/TC1_Test_Specification.md` - TC1 테스트 규격 문서
- `report/TC1_Test_Procedure.md` - TC1 시험 절차 문서
- `report/TC1_Model_and_Data.md` - TC1 모델 및 데이터 문서
- `report/TC1_Operating_Environment.md` - TC1 운영 환경 문서

### 7단계: 인퍼런스 시스템 구축 및 배포

**수행 내용:**
- Whisper-AT 실시간 인퍼런스 파이프라인 개발
- YouTube 영상 자동 처리 시스템 구축
- 세그먼트 단위 처리 로직 구현
- 한국어 레이블 자동 변환 기능 추가

**인퍼런스 주요 기능:**
- 세그먼트 단위 처리 (기본 10초, 조정 가능)
- 6개 목표 한국어 태그 필터링 (박수 소리, 웃음 소리, 기침 소리, 한숨 소리, 환호하는 소리, 박수/환호)
- 시간 해상도 0.4초의 정수배로 자동 조정
- AT 체크포인트 로딩 및 적용
- 결과 JSON/CSV 출력

**관련 저장소:**
- [chat-prompt/whisper-at-demo](https://github.com/chat-prompt/whisper-at-demo) - 인퍼런스 데모 프로젝트

---

## 연구개발 성과

### 정량적 성과

#### A. 모델 성능 지표

| 평가 항목 | 지표 | 값 | 비고 |
|---------|------|-----|------|
| 전체 533개 클래스 | mAP | **0.4148** | 가중치 평균화 적용 |
| AudioSet 527개 클래스 | mAP | 0.4171 | 기존 성능 유지 |
| SONYC-UST 6개 클래스 | mAP | 0.2145 | 신규 클래스 |
| SONYC-UST 16개 주요 클래스 | mAP | **0.4529** | 기존 0.3137 대비 **+44.4%** |

#### B. TC1 테스트 결과 (핵심 성과)

| 음향 종류 | 샘플 수 | 정확도 | 목표 |
|---------|--------|--------|------|
| 박수 소리 | 45개 | 100% | 100% |
| 웃음 소리 | 107개 | 100% | 100% |
| 기침 소리 | 6개 | 100% | 100% |
| 한숨 소리 | 3개 | 100% | 100% |
| 환호하는 소리 | 3개 | 100% | 100% |
| **전체** | **164개** | **100%** | **100%** |

- 음성 인식률(WER) 저하: **없음** (ΔWER ≤ 0.5%p 기준 충족)

#### C. 모델 스펙

| 항목 | 값 |
|------|-----|
| 기본 모델 | Whisper Large-v1 |
| 추가 모듈 | TL-TR (Time and Layer-wise Transformer) |
| 총 클래스 수 | 533개 (AudioSet 527 + SONYC-UST 6) |
| 학습 가능 파라미터 | 약 4,000만 개 |
| 추가 연산량 | < 1% (원본 Whisper 대비) |
| 학습 데이터 | AudioSet 20k + SONYC-UST |
| 학습 Epoch | 50 (가중치 평균화 36-50) |

### 정성적 성과

#### 기술적 성과
1. **원본 Whisper 성능 완벽 유지**
   - ASR 파라미터 동결로 음성 인식 성능 저하 없음
   - API 호환성: 기존 Whisper API와 완벽 호환
   - 최소 계산 비용: 원본 Whisper 대비 1% 미만 추가 연산

2. **도메인 확장 성공**
   - 일반 음향(AudioSet) 527개 클래스 유지
   - 도시 소음(SONYC-UST) 6개 클래스 추가
   - 웨비나/강의 환경 특화 성능 검증

3. **다국어 지원**
   - 한국어 레이블 자동 변환 기능 구현
   - 영어/한국어 seamless 전환

#### 연구적 성과
1. **전이 학습 효과 검증**
   - AudioSet → SONYC-UST 전이 학습 성공
   - 도메인 간 지식 전이 가능성 입증

2. **실시간 처리 가능성**
   - 세그먼트 단위 인퍼런스로 긴 영상 처리 가능
   - 시간 해상도 0.4초 단위 조정 가능

#### 응용적 성과
1. **콘텐츠 리퍼포징**
   - 웨비나/강의 영상 자동 분석 가능
   - 하이라이트 구간 자동 추출 기반 마련

2. **멀티모달 분석**
   - 음성 + 음향 동시 인식으로 콘텐츠 이해도 향상
   - 청중 반응 분석 가능

3. **실용화 준비**
   - TC1 테스트 통과로 실제 서비스 적용 가능
   - PyPI 패키지 배포 완료 (whisper-at 0.6)

### 목표 달성도 평가

| 연구 목표 | 목표치 | 달성치 | 달성도 |
|---------|--------|--------|--------|
| SONYC-UST 클래스 성능 향상 | mAP 향상 | +44.4% | **초과 달성** |
| 웨비나/강의 음향 인식 | 100% 정확도 | 100% | **목표 달성** |
| 음성 인식 성능 유지 | ΔWER ≤ 0.5%p | 저하 없음 | **목표 달성** |
| 실시간 처리 가능성 | 구현 | 세그먼트 처리 구현 | **목표 달성** |

---

## 설치 및 사용법

### 설치

```bash
# PyPI에서 설치 (권장)
pip install whisper-at

# Mac/Windows 사용자
pip install numba numpy torch tqdm more-itertools tiktoken==0.3.3
pip install --no-deps whisper-at
```

### 기본 사용법

```python
import whisper_at as whisper

# 모델 로드 (533개 클래스 지원)
model = whisper.load_model("large-v1")

# 음성 인식 + 음향 태깅 동시 수행
result = model.transcribe("audio.mp3", at_time_res=10)

# ASR 결과
print(result["text"])

# 음향 태깅 결과
audio_tag_result = whisper.parse_at_label(
    result,
    language='follow_asr',
    top_k=5,
    p_threshold=-1,
    include_class_list=list(range(533))  # 533개 클래스 전체
)
print(audio_tag_result)
```

### 결합 학습 실행

```bash
cd src/whisper_at_train
./run_combined_training.sh
```

### 모델 평가

```bash
python src/whisper_at_train/evaluate_pretrained_whisper_at.py
```

---

## 프로젝트 구조

```
whisper-at/
├── README.md                          # 본 문서
├── README_ORIGINAL.md                 # 원본 Whisper-AT README
├── CLAUDE.md                          # Claude Code 개발 가이드
├── report/                            # 보고서 및 문서
│   ├── TIPS_Final_Report_Material.md  # TIPS 최종 보고서 작성 자료
│   ├── whisper_at_report.md           # 종합 연구개발 결과 보고서
│   ├── TC1_Test_Specification.md      # TC1 시험 규격서
│   ├── TC1_Test_Procedure.md          # TC1 시험 절차서
│   ├── TC1_Model_and_Data.md          # TC1 모델 및 데이터 요약
│   ├── TC1_Operating_Environment.md   # TC1 운영 환경
│   └── TC1_Test_Environment_Diagram.md # TC1 시험 환경 구성도
├── src/whisper_at_train/              # 학습 코드
│   ├── run_combined_training.sh       # 결합 학습 스크립트 (신규)
│   ├── run_as_sonyc.sh                # SONYC 학습 스크립트 (신규)
│   ├── run_as_full_train.sh           # AudioSet 학습 스크립트
│   ├── run.py                         # 학습 메인 진입점
│   ├── traintest.py                   # 학습/테스트 핵심 로직
│   ├── models.py                      # TL-TR 모델 아키텍처
│   ├── dataloader_feat.py             # 특징 기반 데이터로더
│   └── evaluate_pretrained_whisper_at.py  # 모델 평가 스크립트
├── data/processed_data/               # 전처리된 데이터 (신규)
│   ├── class_labels_indices_extended.csv  # 533개 클래스 정의
│   ├── sonyc_new_class_mapping.json   # SONYC 6개 신규 클래스 매핑
│   ├── combined_train.json            # 결합 학습 데이터
│   ├── combined_val.json              # 결합 검증 데이터
│   └── sonyc_*.json                   # SONYC-UST 데이터셋
├── package/whisper-at/                # PyPI 패키지
│   └── whisper_at/
│       ├── __init__.py                # 모델 로딩 (체크포인트 URL 업데이트)
│       ├── transcribe.py              # 트랜스크립션 API
│       └── at_post_processing.py      # 음향 태깅 후처리
└── sample/                            # 샘플 코드 및 데모
    └── whisper_at_demo.ipynb          # Jupyter 노트북 데모
```

---

## 추가/수정된 파일 목록

### 데이터 파일 (신규)
```
data/processed_data/
├── audioset_train.json              # AudioSet 20k 학습 데이터
├── audioset_val.json                # AudioSet 검증 데이터
├── class_labels_indices_extended.csv # 533개 클래스 정의 (527 + 6)
├── combined_train.json              # AudioSet + SONYC-UST 결합 학습 데이터
├── combined_val.json                # 결합 검증 데이터
├── sonyc_new_class_mapping.json     # SONYC 6개 신규 클래스 매핑
├── sonyc_new_train.json             # SONYC 신규 클래스 학습 데이터
├── sonyc_new_val.json               # SONYC 신규 클래스 검증 데이터
├── sonyc_new_test.json              # SONYC 신규 클래스 테스트 데이터
├── sonyc_new_train_weight.csv       # SONYC 신규 클래스 학습 가중치
├── sonyc_train.json                 # SONYC 전체 학습 데이터
├── sonyc_val.json                   # SONYC 전체 검증 데이터
├── sonyc_test.json                  # SONYC 전체 테스트 데이터
├── sonyc_test_filtered.json         # 필터링된 테스트 데이터
├── sonyc_test_filtered_ohe.json     # 원핫 인코딩된 테스트 데이터
└── sonyc_train_weight.csv           # 학습 가중치

data/
├── youtube_*.csv                    # TC1 테스트 데이터 (10개 영상)
```

### 학습 코드 (수정/신규)
```
src/whisper_at_train/
├── run_combined_training.sh         # 결합 학습 스크립트 (신규)
├── run_as_sonyc.sh                  # SONYC 학습 스크립트 (신규)
├── evaluate_pretrained_whisper_at.py # 모델 평가 스크립트 (신규)
├── run.py                           # 학습 메인 (수정)
├── traintest.py                     # 학습/테스트 로직 (수정)
└── dataloader_feat.py               # 데이터로더 (수정)
```

### 데이터 처리 스크립트 (신규)
```
script/
├── convert_audioset_json.py         # AudioSet JSON 변환
├── convert_sonyc_ust_to_ohe.py      # SONYC 원핫 인코딩 변환
├── extract_hf_audio_by_index.py     # HuggingFace 오디오 추출
├── extract_hf_audioset.py           # HuggingFace AudioSet 추출
├── extract_sonyc_features.py        # SONYC 특징 추출
├── filter_sonyc_labels.py           # SONYC 레이블 필터링
├── find_small_files.py              # 소용량 파일 탐색
├── match_labels_audioset_sonyc.py   # AudioSet-SONYC 레이블 매칭
├── merge_audioset_sonyc_json.py     # AudioSet-SONYC JSON 병합
├── process_sonyc_new_classes_only.py # SONYC 신규 클래스 처리
├── process_sonyc_ust_csv.py         # SONYC CSV 처리
└── semantic_sonyc_to_audioset_mapping.json # 시맨틱 매핑
```

### PyPI 패키지 (수정)
```
package/whisper-at/whisper_at/
├── __init__.py                      # 모델 로딩 및 체크포인트 URL (수정)
├── transcribe.py                    # 트랜스크립션 API (수정)
├── model.py                         # 모델 정의 (수정)
├── version.py                       # 버전 0.6 (수정)
└── assets/label_name_dict.json      # 레이블 사전 (수정)
```

### 문서 (신규)
```
report/
├── TIPS_Final_Report_Material.md    # TIPS 최종 보고서 작성 자료
├── whisper_at_report.md             # 종합 연구개발 결과 보고서
├── TC1_Test_Specification.md        # TC1 시험 규격서
├── TC1_Test_Procedure.md            # TC1 시험 절차서
├── TC1_Model_and_Data.md            # TC1 모델 및 데이터 요약
├── TC1_Operating_Environment.md     # TC1 운영 환경
├── TC1_Test_Environment_Diagram.md  # TC1 시험 환경 구성도
└── images/                          # 보고서 이미지
```

### 프로젝트 설정 (신규/수정)
```
├── CLAUDE.md                        # Claude Code 개발 가이드 (신규)
├── pyproject.toml                   # Poetry 프로젝트 설정 (신규)
├── poetry.lock                      # 의존성 잠금 파일 (신규)
├── .gitignore                       # Git 무시 파일 (수정)
└── notebook/training_stat_viewer.ipynb # 학습 통계 뷰어 (수정)
```

---

## 향후 활용 계획

### 기술 고도화
- 더 많은 도메인 특화 클래스 추가
- 실시간 처리 성능 최적화
- 모바일/엣지 디바이스 포팅

### 서비스 적용
- 웨비나/온라인 강의 플랫폼 통합
- 콘텐츠 리퍼포징 자동화 서비스
- 멀티모달 콘텐츠 분석 솔루션

### 연구 확장
- 다른 ASR 모델과의 통합
- 추가 음향 이벤트 클래스 확장
- 다국어 음향 레이블 지원 강화

---

## 참고 문헌

### 원본 논문
- [Whisper-AT: Noise-Robust Automatic Speech Recognizers are Also Strong Audio Event Taggers](https://arxiv.org/pdf/2307.03183.pdf) (Interspeech 2023)
- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/pdf/2212.04356.pdf) (OpenAI)

### 인용
```bibtex
@inproceedings{gong_whisperat,
  author={Gong, Yuan and Khurana, Sameer and Karlinsky, Leonid and Glass, James},
  title={Whisper-AT: Noise-Robust Automatic Speech Recognizers are Also Strong Audio Event Taggers},
  year=2023,
  booktitle={Proc. Interspeech 2023}
}
```

---

## 관련 저장소

- **원본 Whisper-AT**: [YuanGongND/whisper-at](https://github.com/YuanGongND/whisper-at)
- **인퍼런스 데모**: [chat-prompt/whisper-at-demo](https://github.com/chat-prompt/whisper-at-demo)

---

## 라이선스

본 프로젝트는 원본 Whisper-AT와 동일하게 BSD 라이선스를 따릅니다. 상업적 사용이 허용됩니다.

---

## 연락처

- **본 프로젝트 (Fork)**: [chat-prompt/whisper-at](https://github.com/chat-prompt/whisper-at)
- **원본 프로젝트**: Yuan Gong - [yuangong@mit.edu](mailto:yuangong@mit.edu)
