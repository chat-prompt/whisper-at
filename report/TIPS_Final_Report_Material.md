# TIPS 종합 음향 인식 기술 개발 연구과제 최종 보고서 작성 자료

## 연구개발 전체 과정

### 1. 기반 연구 및 기술 분석

**수행 내용:**
- OpenAI Whisper 및 Whisper-AT 핵심 논문 심층 분석
- 음성 인식과 음향 태깅 동시 수행 가능성 검증
- 공식 GitHub 저장소(YuanGongND/whisper-at) 코드 분석
- 기술적 타당성 검토 및 연구 방향 수립

**관련 자료:**
- `2212.04356v1.pdf` - Whisper: Robust Speech Recognition via Large-Scale Weak Supervision (OpenAI 논문)
- `2307.03183v1.pdf` - Whisper-AT: Noise-Robust Automatic Speech Recognizers are Also Strong General Audio Event Taggers (원본 논문)
- `Whisper_논문_리뷰.pdf` - Whisper 논문 상세 리뷰 문서
- `Whisper-AT_논문_리뷰.pdf` - Whisper-AT 논문 상세 리뷰 문서
- GitHub 공식 저장소: https://github.com/YuanGongND/whisper-at

**주요 발견:**
- Whisper는 노이즈에 강인하지만 noise-invariant가 아닌 noise-variant 표현을 학습
- 원본 Whisper 파라미터를 동결하고 TL-TR(Time and Layer-wise Transformer)만 학습하여 계산 비용 최소화
- AudioSet 527개 클래스에 대해 mAP 0.404~0.421 성능 달성 가능

---

### 2. 자체 데이터셋 구축 시도 및 전략 전환

**수행 내용:**
- 웨비나/온라인 스터디 환경 특화 음향 데이터 수집
- 데이터 레이블링 프로세스 설계 및 품질 관리 시도
- 데이터셋 구축의 현실적 한계 분석
- 공개 데이터셋 활용으로 연구 전략 변경

**관련 자료:**
- 자체 수집 데이터 (비공개, 품질 이슈로 최종 미사용)
- 데이터셋 구축 과정 내부 문서

**직면한 과제:**
- 데이터 다양성 및 품질 확보의 어려움
- 레이블링 복잡성 및 시간 소요
- 데이터 편향성 문제
- 대규모 고품질 데이터셋 구축의 현실적 제약

**전략적 결정:**
- SONYC-UST 공개 데이터셋 활용으로 방향 전환
- 검증된 데이터셋을 통한 연구 효율성 확보

---

### 3. SONYC-UST 데이터셋 통합 및 모델 확장

**수행 내용:**
- SONYC-UST (Sounds of New York City - Urban Sound Tagging) 데이터셋 분석
- 6개 신규 도시 소음 클래스 정의 및 레이블 체계 구축
- AudioSet 527개 + SONYC-UST 6개 = 총 533개 클래스로 확장
- 클래스 레이블 매핑 파일 생성 및 데이터 전처리

**관련 자료:**
- `whisper-at/data/processed_data/audioset_train.json` - AudioSet 20k 학습 데이터
- `whisper-at/data/processed_data/sonyc_new_train.json` - SONYC-UST 학습 데이터
- `whisper-at/data/processed_data/combined_train.json` - AudioSet + SONYC-UST 결합 학습 데이터
- `whisper-at/data/processed_data/combined_val.json` - 결합 검증 데이터
- `whisper-at/data/processed_data/class_labels_indices_extended.csv` - 533개 클래스 정의 파일 (534줄, 헤더 포함)
- `whisper-at/data/processed_data/sonyc_new_class_mapping.json` - SONYC 6개 신규 클래스 매핑

**신규 SONYC-UST 6개 클래스:**
1. Amplified speech (증폭된 음성) - index 527
2. Hoe ram (굴착기 파쇄기) - index 528
3. Large rotating saw (대형 회전톱) - index 529
4. Non machinery impact (비기계적 충격음) - index 530
5. Pile driver (항타기) - index 531
6. Small medium rotating saw (중소형 회전톱) - index 532

---

### 4. 모델 파인튜닝 및 학습

**수행 내용:**
- Whisper-AT Large-v1 기반 TL-TR 모델 파인튜닝
- AudioSet 20k + SONYC-UST 결합 데이터셋으로 학습
- 가중치 평균화(Weight Averaging) 기법 적용 (epoch 36-50)
- 하이퍼파라미터 튜닝 및 최적화

**관련 코드:**
- `whisper-at/src/whisper_at_train/run_combined_training.sh` - 결합 학습 실행 스크립트
- `whisper-at/src/whisper_at_train/run.py` - 학습 메인 진입점
- `whisper-at/src/whisper_at_train/traintest.py` - 학습/테스트 핵심 로직
- `whisper-at/src/whisper_at_train/models.py` - TL-TR 모델 아키텍처 정의
- `whisper-at/src/whisper_at_train/dataloader_feat.py` - 특징 기반 데이터로더

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

---

### 5. 모델 평가 및 성능 검증

**수행 내용:**
- 일반 음향 태깅 성능 평가 (AudioSet 527개 클래스)
- SONYC-UST 특정 클래스 성능 비교 분석
- 원본 Whisper-AT 모델 대비 성능 개선 측정

**관련 코드:**
- `whisper-at/src/whisper_at_train/evaluate_pretrained_whisper_at.py` - 사전학습 모델 평가 스크립트

**정량적 성과:**

**A. 전체 533개 클래스 성능:**
- 전체 mAP: **0.4148** (가중치 평균화 적용)
- AudioSet 527개 클래스 mAP: 0.4171
- SONYC-UST 6개 클래스 mAP: 0.2145

**B. SONYC-UST 16개 주요 클래스 성능 향상:**
- 기존 Whisper-AT 논문 모델: mAP **0.3137**
- 본 연구 파인튜닝 모델: mAP **0.4529**
- **성능 향상: +44.4%** (절대값 0.1392 증가)

---

### 6. TC1 테스트셋 구축 및 검증

**수행 내용:**
- 웨비나/강의 환경 특화 테스트 데이터셋 구축
- YouTube 웨비나/강의 영상 10개 선정 및 레이블링
- 5개 목표 음향 정확도 평가 프로토콜 수립
- 음성 인식 성능(WER) 측정

**관련 자료:**
- `whisper-at/report/TC1_Test_Specification.md` - TC1 테스트 규격 문서
- `whisper-at/report/TC1_Test_Procedure.md` - TC1 시험 절차 문서
- `whisper-at/report/TC1_Model_and_Data.md` - TC1 모델 및 데이터 문서
- `whisper-at/report/TC1_Operating_Environment.md` - TC1 운영 환경 문서
- TC1 테스트 데이터셋 (YouTube 웨비나 10개, 총 164개 음향 샘플)

**TC1 테스트 규격:**
- **목표**: 30dB SNR 웨비나/강의 영상에서 5가지 음향(박수, 웃음, 기침, 한숨, 환호) 100% 정확도 검출
- **평가 기준**: 음향 인식 정확도 = (TP + TN) / (TP + TN + FP + FN) × 100
- **허용 오차**: 실제 음향 발생 시점 ±1초 이내 태그하면 "검출"로 판정
- **추가 조건**: 음성 인식 성능 저하 없음 (ΔWER ≤ 0.5%p)

**테스트 데이터셋 구성:**
- 박수 소리: 45개 샘플 (9개 파일)
- 웃음 소리: 107개 샘플 (5개 파일)
- 기침 소리: 6개 샘플 (1개 파일)
- 한숨 소리: 3개 샘플 (1개 파일)
- 환호하는 소리: 3개 샘플 (1개 파일)

**TC1 테스트 결과:**
- 5개 목표 음향 인식 정확도: **100%** ✓
- 음성 인식률(WER) 저하 없음 ✓
- **테스트 통과**

---

### 7. 인퍼런스 시스템 구축 및 배포

**수행 내용:**
- Whisper-AT 실시간 인퍼런스 파이프라인 개발
- YouTube 영상 자동 처리 시스템 구축
- 세그먼트 단위 처리 로직 구현
- 한국어 레이블 자동 변환 기능 추가

**관련 코드 (whisper-at-demo 디렉토리):**
- `whisper-at-demo/run_whisper_at.py` - MP4 비디오에서 오디오 태그 추출 메인 스크립트
- `whisper-at-demo/inference_youtube.py` - YouTube 영상 직접 인퍼런스
- `whisper-at-demo/inference_youtube_by_segments.py` - 세그먼트 단위 인퍼런스
- `whisper-at-demo/inference_wav.py` - WAV 파일 직접 인퍼런스
- `whisper-at-demo/eval_audio_tags.py` - 오디오 태그 정확도 평가
- `whisper-at-demo/download_youtube_to_wavs.py` - YouTube 영상을 WAV 파일로 변환

**인퍼런스 주요 기능:**
- 세그먼트 단위 처리 (기본 10초, 조정 가능)
- 6개 목표 한국어 태그 필터링 (박수 소리, 웃음 소리, 기침 소리, 한숨 소리, 환호하는 소리, 박수/환호)
- 시간 해상도 0.4초의 정수배로 자동 조정
- AT 체크포인트 로딩 및 적용
- 결과 JSON/CSV 출력

---

## 연구개발과제 보고서 작성 내용

### 1. 연구개발과제의 개요

**과제명:** 종합 음향 인식 기술 개발

**연구 목표:**
- OpenAI Whisper 기반 음성 인식 성능을 유지하면서 음향 이벤트 태깅 기능 강화
- 도시 소음 환경(SONYC-UST)에 대한 음향 인식 성능 향상
- 웨비나/강의 환경에서 5가지 핵심 음향(박수, 웃음, 기침, 한숨, 환호) 100% 정확도 달성
- 음성 인식 성능 저하 없이 음향 태깅 기능 통합

**연구 필요성:**
- 온라인 콘텐츠 자동 분석 및 리퍼포징 수요 증가
- 음성뿐만 아니라 음향 정보를 활용한 멀티모달 분석 필요
- 웨비나, 온라인 강의 등에서 청중 반응 자동 분석 요구

---

### 2. 연구개발과제의 수행 과정 및 수행 내용

#### 2.1 기반 연구 단계

**문헌 조사 및 기술 분석:**
- OpenAI Whisper 논문 분석: 680,000시간의 다국어 음성 데이터로 학습된 robust한 ASR 모델
- Whisper-AT 논문 분석: Whisper의 노이즈 강인성을 활용하여 음향 태깅 기능 추가
- TL-TR(Time and Layer-wise Transformer) 아키텍처 이해

**핵심 발견:**
- Whisper는 noise-variant 표현을 학습하여 노이즈에 강인함
- 원본 Whisper 파라미터 동결로 ASR 성능 유지 가능
- 추가 연산량 1% 미만으로 음향 태깅 기능 구현 가능

#### 2.2 데이터셋 구축 단계

**자체 데이터셋 구축 시도:**
- 웨비나/온라인 스터디 환경 특화 음향 데이터 수집
- 레이블링 프로세스 설계 및 품질 관리 시도

**과제 및 전략 변경:**
- 데이터 다양성 및 품질 확보의 어려움 직면
- SONYC-UST 공개 데이터셋 활용으로 전략 변경
- 검증된 데이터셋을 통한 연구 효율성 확보

#### 2.3 모델 확장 단계

**SONYC-UST 통합:**
- 6개 신규 도시 소음 클래스 추가 정의
- AudioSet 527개 + SONYC-UST 6개 = 총 533개 클래스로 확장
- 클래스 레이블 매핑 체계 구축
- 데이터 전처리 및 결합 데이터셋 생성

#### 2.4 모델 학습 단계

**파인튜닝 수행:**
- Whisper-AT Large-v1 기반 TL-TR 모델 학습
- AudioSet 20k + SONYC-UST 결합 데이터셋 활용
- 50 epoch 학습 후 36-50 epoch 가중치 평균화 적용
- 하이퍼파라미터 최적화 (learning rate, mixup, label smoothing 등)

**학습 환경:**
- Google Cloud Platform GPU VM (NVIDIA Tesla T4)
- CUDA 12.4, PyTorch 2.3.1
- 약 4,000만 개 학습 가능 파라미터

#### 2.5 평가 및 검증 단계

**성능 평가:**
- AudioSet 527개 클래스: mAP 0.4171
- SONYC-UST 16개 주요 클래스: mAP 0.4529 (기존 대비 +44.4%)
- TC1 테스트: 5개 목표 음향 100% 정확도 달성

#### 2.6 시스템 구축 단계

**인퍼런스 파이프라인 개발:**
- YouTube 영상 자동 처리 시스템
- 세그먼트 단위 실시간 처리
- 한국어 레이블 자동 변환

---

### 3. 연구개발과제의 수행 결과 및 목표 달성 정도

#### 3.1 연구수행 결과

#### (1) 정성적 연구개발성과

**A. 기술적 성과**

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

**B. 연구적 성과**

1. **전이 학습 효과 검증**
   - AudioSet → SONYC-UST 전이 학습 성공
   - 도메인 간 지식 전이 가능성 입증

2. **실시간 처리 가능성**
   - 세그먼트 단위 인퍼런스로 긴 영상 처리 가능
   - 시간 해상도 0.4초 단위 조정 가능

**C. 응용적 성과**

1. **콘텐츠 리퍼포징**
   - 웨비나/강의 영상 자동 분석 가능
   - 하이라이트 구간 자동 추출 기반 마련

2. **멀티모달 분석**
   - 음성 + 음향 동시 인식으로 콘텐츠 이해도 향상
   - 청중 반응 분석 가능

3. **실용화 준비**
   - TC1 테스트 통과로 실제 서비스 적용 가능
   - PyPI 패키지 배포 완료 (whisper-at 0.6)

---

#### (2) 정량적 연구개발성과

**A. 모델 성능 지표**

| 평가 항목 | 지표 | 값 | 비고 |
|---------|------|-----|------|
| 전체 533개 클래스 mAP | mAP | 0.4148 | 가중치 평균화 적용 |
| AudioSet 527개 클래스 | mAP | 0.4171 | 기존 성능 유지 |
| SONYC-UST 6개 클래스 | mAP | 0.2145 | 신규 클래스 |
| SONYC-UST 16개 주요 클래스 | mAP | **0.4529** | 기존 0.3137 대비 **+44.4%** |

**B. TC1 테스트 결과 (핵심 성과)**

| 음향 종류 | 샘플 수 | 정확도 | 목표 |
|---------|--------|--------|------|
| 박수 소리 | 45개 | 100% | 100% |
| 웃음 소리 | 107개 | 100% | 100% |
| 기침 소리 | 6개 | 100% | 100% |
| 한숨 소리 | 3개 | 100% | 100% |
| 환호하는 소리 | 3개 | 100% | 100% |
| **전체** | **164개** | **100%** | **100%** ✓ |

- **음성 인식률(WER) 저하**: 없음 (ΔWER ≤ 0.5%p 기준 충족) ✓

**C. 모델 스펙**

| 항목 | 값 |
|------|-----|
| 기본 모델 | Whisper Large-v1 |
| 추가 모듈 | TL-TR (Time and Layer-wise Transformer) |
| 총 클래스 수 | 533개 (AudioSet 527 + SONYC-UST 6) |
| 학습 가능 파라미터 | 약 4,000만 개 |
| 추가 연산량 | < 1% (원본 Whisper 대비) |
| 학습 데이터 | AudioSet 20k + SONYC-UST |
| 학습 Epoch | 50 (가중치 평균화 36-50) |

**D. 시스템 성능**

| 항목 | 값 |
|------|-----|
| 처리 단위 | 세그먼트 (기본 10초) |
| 시간 해상도 | 0.4초의 정수배 |
| 지원 언어 | 다국어 (한국어 레이블 지원) |
| 배포 형태 | PyPI 패키지 (whisper-at 0.6) |

---

### 목표 달성도 평가

| 연구 목표 | 목표치 | 달성치 | 달성도 |
|---------|--------|--------|--------|
| SONYC-UST 클래스 성능 향상 | mAP 향상 | +44.4% | **초과 달성** |
| 웨비나/강의 음향 인식 | 100% 정확도 | 100% | **목표 달성** |
| 음성 인식 성능 유지 | ΔWER ≤ 0.5%p | 저하 없음 | **목표 달성** |
| 실시간 처리 가능성 | 구현 | 세그먼트 처리 구현 | **목표 달성** |

**종합 평가:** 모든 핵심 목표를 달성하였으며, 특히 SONYC-UST 클래스 성능은 목표를 초과 달성함

---

## 자료 목록

### 논문 및 문헌
1. `2212.04356v1.pdf` - Whisper 원본 논문
2. `2307.03183v1.pdf` - Whisper-AT 원본 논문
3. `Whisper_논문_리뷰.pdf` - Whisper 논문 리뷰
4. `Whisper-AT_논문_리뷰.pdf` - Whisper-AT 논문 리뷰

### 최종 보고서
5. `I._종합_음향_인식_기술_개발.pdf` - 음향인식 기술 개발 결과 보고서

### 데이터 및 코드
6. `whisper-at.tar.gz` - 학습 코드 및 데이터셋 전체 (src/whisper_at_train, data/processed_data 포함)
7. `whisper-at-demo.tar.gz` - 인퍼런스 코드 전체 (YouTube 영상 처리 및 평가 코드)

### GitHub 저장소
8. 본 프로젝트 GitHub: https://github.com/chat-prompt/whisper-at
9. 원본 Whisper-AT GitHub: https://github.com/YuanGongND/whisper-at
10. 인퍼런스 프로젝트 GitHub: https://github.com/chat-prompt/whisper-at-demo

---

## 프로젝트 구조 요약

```
whisper-at/
├── report/                        # 보고서 및 문서
│   ├── whisper_at_report.md      # 최종 연구개발 결과 보고서
│   ├── TC1_Test_Specification.md
│   ├── TC1_Test_Procedure.md
│   ├── TC1_Model_and_Data.md
│   └── TC1_Operating_Environment.md
├── src/whisper_at_train/          # 학습 코드
│   ├── run_combined_training.sh   # 결합 학습 스크립트
│   ├── run.py                     # 학습 메인
│   ├── traintest.py               # 학습/테스트 로직
│   ├── models.py                  # TLTR 모델
│   ├── dataloader_feat.py         # 데이터로더
│   └── evaluate_pretrained_whisper_at.py
├── data/processed_data/           # 전처리된 데이터
│   ├── combined_train.json        # 결합 학습 데이터
│   ├── combined_val.json          # 결합 검증 데이터
│   ├── class_labels_indices_extended.csv  # 533개 클래스
│   └── sonyc_new_class_mapping.json       # 6개 신규 클래스
└── package/whisper-at/            # PyPI 패키지

whisper-at-demo/                   # 인퍼런스 데모
├── run_whisper_at.py              # MP4→태그 추출
├── inference_youtube.py           # YouTube 인퍼런스
├── eval_audio_tags.py             # 평가
└── data/                          # 테스트 데이터 및 결과
```

---

## 핵심 성과 요약

### 연구적 성과
- SONYC-UST 16개 주요 클래스 mAP 44.4% 향상
- AudioSet → SONYC-UST 전이 학습 검증
- 음성 인식 + 음향 태깅 동시 수행 성공

### 기술적 성과
- TC1 테스트 100% 정확도 달성
- 음성 인식 성능 저하 없음 (ΔWER = 0)
- 추가 연산량 < 1%

### 응용적 성과
- PyPI 패키지 배포 (whisper-at 0.6)
- 실시간 인퍼런스 파이프라인 구축
- 웨비나/강의 영상 자동 분석 가능

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
