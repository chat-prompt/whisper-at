# Whisper-AT Extended: SONYC-UST 통합 및 웨비나 음향 인식

> **원본 프로젝트**: [YuanGongND/whisper-at](https://github.com/YuanGongND/whisper-at)
> **본 저장소**: [chat-prompt/whisper-at](https://github.com/chat-prompt/whisper-at)

## 프로젝트 개요

이 프로젝트는 [Whisper-AT](https://github.com/YuanGongND/whisper-at) 논문 구현체를 fork하여 **TIPS 종합 음향 인식 기술 개발 연구과제**를 수행한 결과물입니다.

### 연구 목표
- OpenAI Whisper 기반 음성 인식 성능을 유지하면서 음향 이벤트 태깅 기능 강화
- SONYC-UST 도시 소음 데이터셋 통합으로 클래스 확장 (527 → 533 클래스)
- 웨비나/강의 환경에서 5가지 핵심 음향(박수, 웃음, 기침, 한숨, 환호) 100% 정확도 달성
- 음성 인식 성능 저하 없이 음향 태깅 기능 통합

### 핵심 성과
| 평가 항목 | 결과 |
|---------|------|
| 전체 533개 클래스 mAP | **0.4148** |
| SONYC-UST 16개 주요 클래스 | **mAP 0.4529** (기존 대비 +44.4%) |
| TC1 테스트 5개 목표 음향 정확도 | **100%** |
| 음성 인식률(WER) 저하 | **없음** |

---

## 원본 대비 주요 변경 사항

### 1. SONYC-UST 데이터셋 통합 및 클래스 확장

**신규 6개 도시 소음 클래스 추가:**
| Index | 클래스명 | 설명 |
|-------|---------|------|
| 527 | Amplified speech | 증폭된 음성 |
| 528 | Hoe ram | 굴착기 파쇄기 |
| 529 | Large rotating saw | 대형 회전톱 |
| 530 | Non machinery impact | 비기계적 충격음 |
| 531 | Pile driver | 항타기 |
| 532 | Small medium rotating saw | 중소형 회전톱 |

**추가된 데이터 파일:**
- `data/processed_data/class_labels_indices_extended.csv` - 533개 클래스 정의
- `data/processed_data/sonyc_new_class_mapping.json` - SONYC 6개 신규 클래스 매핑
- `data/processed_data/combined_train.json` - AudioSet + SONYC-UST 결합 학습 데이터
- `data/processed_data/combined_val.json` - 결합 검증 데이터
- `data/processed_data/sonyc_*.json` - SONYC-UST 데이터셋 파일들

### 2. 학습 스크립트 추가/수정

**새로 추가된 학습 스크립트:**
- `src/whisper_at_train/run_combined_training.sh` - AudioSet + SONYC-UST 결합 학습
- `src/whisper_at_train/run_as_sonyc.sh` - SONYC-UST 단독 학습

**학습 하이퍼파라미터:**
```
Model: Whisper Large-v1 + TL-TR (Time and Layer-wise Transformer)
Learning Rate: 1e-6 (초기), ReduceLROnPlateau (patience=2, decay=0.75)
Batch Size: 48
Epochs: 50
Weight Averaging: epoch 36-50
학습 가능 파라미터: 약 4,000만 개
```

### 3. 모델 체크포인트 업데이트

파인튜닝된 모델 체크포인트가 Dropbox에 업로드되어 자동 다운로드됩니다:
- `package/whisper-at/whisper_at/__init__.py` - 모델 URL 업데이트

### 4. TC1 테스트 환경 구축

**테스트 규격:**
- 30dB SNR 조건의 웨비나/강의 영상 10편
- 5가지 목표 음향: 박수, 웃음, 기침, 한숨, 환호
- 평가 기준: 실제 음향 발생 시점 ±1초 이내 검출

**테스트 데이터셋 (164개 샘플):**
- 박수 소리: 45개 (9개 파일)
- 웃음 소리: 107개 (5개 파일)
- 기침 소리: 6개 (1개 파일)
- 한숨 소리: 3개 (1개 파일)
- 환호하는 소리: 3개 (1개 파일)

### 5. 문서화

**추가된 문서 (`report/` 디렉토리):**
- `TC1_Test_Specification.md` - TC1 시험 규격서
- `TC1_Test_Procedure.md` - TC1 시험 절차서
- `TC1_Model_and_Data.md` - TC1 모델 및 데이터 요약
- `TC1_Operating_Environment.md` - TC1 운영 환경
- `TC1_Test_Environment_Diagram.md` - TC1 시험 환경 구성도
- `whisper_at_report.md` - 종합 연구개발 결과 보고서
- `TIPS_Final_Report_Material.md` - TIPS 최종 보고서 작성 자료

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
├── report/                            # 연구 보고서 및 문서
│   ├── TC1_Test_Specification.md      # TC1 시험 규격서
│   ├── TC1_Test_Procedure.md          # TC1 시험 절차서
│   ├── TC1_Model_and_Data.md          # TC1 모델 및 데이터 요약
│   ├── TC1_Operating_Environment.md   # TC1 운영 환경
│   ├── whisper_at_report.md           # 종합 연구개발 결과 보고서
│   └── TIPS_Final_Report_Material.md  # TIPS 최종 보고서 자료
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

## 운영 환경

**TC1 테스트 환경:**
- OS: Debian GNU/Linux 11 (Kernel 5.10.0-35-cloud-amd64)
- GPU: NVIDIA Tesla T4 (15GB VRAM)
- CUDA: 12.4
- Python: 3.10.17
- PyTorch: 2.3.1
- whisper-at: 0.6

---

## Git 커밋 히스토리 주요 변경 사항

### 데이터셋 및 클래스 확장
- `224fa06` - SONYC-UST 데이터셋 파일 추가 (class_labels, class_mapping, train/val/test JSON)
- `1f40285` - SONYC-UST 필터링 데이터 및 원핫 인코딩 레이블 추가
- `6a9316f` - SONYC-UST 데이터셋 및 학습 스크립트 업데이트
- `1f5c09e` - 클래스 레이블 및 모델 설정 업데이트 (533 클래스)

### 학습 스크립트 및 모델
- `270f4ac` - SONYC-UST CSV 어노테이션 처리 스크립트 추가
- `6457761` - SONYC-UST용 Whisper 특징 추출 스크립트 추가
- `9ed6923` - 모델 URL 업데이트 및 SONYC 학습 스크립트 추가
- `35aae69` - 학습 스크립트 및 모델 파라미터 업데이트
- `7447a37` - 결합 학습 스크립트 개선

### 모델 및 추론 개선
- `5526092` - 모델 호환성을 위한 오디오 태그 차원 업데이트
- `5ff9cd8` - AT 모델 체크포인트 로딩 디버깅 및 키 수정
- `107c30b` - load_model 함수에 low-compute 모델 지원 추가
- `f0a62c7` - 버전 0.6으로 업데이트
- `ba220ab` - 오디오 감지를 위한 음성 존재 임계값 업데이트
- `de19c79` - transcribe.py에 SONYC 클래스 억제 휴리스틱 추가

### 문서화 및 테스트
- `d2d29ad` - TC1 시험 규격서 추가
- `ad19d69` - TC1 시험 절차서 추가
- `e95fbc3` - TC1 운영환경 및 시험 환경 구성도 문서 추가
- `10d53d4` - TC1 모델 및 데이터 요약 문서 추가
- `3ed677c` - CLAUDE.md 파일 추가

### CI/CD 및 개발 환경
- `60e9a1e` - 프로젝트 설정 및 의존성 추가 (pyproject.toml)
- `a50982f` - Python 버전 요구사항 3.11 → 3.10 변경

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
