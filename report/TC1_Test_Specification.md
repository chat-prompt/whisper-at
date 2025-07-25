# TC1(음성 외 음향 인식률) 시험 규격서

## 1. 시험 항목
- **음성 외 음향 인식률**

## 2. 시험 목표 및 기준

### <시험목표>
- 30 dB SNR 조건의 웨비나·강의 영상 10편에서 5가지 주요 음향(박수, 웃음, 기침, 한숨, 환호)을 **100 % 정확도**로 검출·분류하면서, 기존 Whisper Large-v1 대비 **음성 인식률 저하(Δ WER) 0.5 %p 이하**를 달성하는지 확인한다.

### <시험기준>
1. **음향 인식 정확도(Accuracy)**  
   - 목표 음향 *TP·TN·FP·FN* 을 이용한 정확도 산정  
   - 평가 조건: 모델이 실제 음향 발생 시점 ±1 초 이내에 태그하면 "검출"으로 판정
2. **음성 인식률(Word Error Rate, WER)**  
   - 기준 모델(Whisper Large-v1)과 시험 모델(Whisper-AT) 간 WER 차이(Δ WER) 산정  
   - Δ WER ≤ **0.5 %p** 이면 음성 인식 성능 저하 없음으로 판정

### 산정식
- Accuracy : `Accuracy = (TP + TN) / (TP + TN + FP + FN) × 100`
  - TP: 목표 음향 포함 & 검출
  - TN: 미포함 & 미검출
  - FP: 미포함 & 검출
  - FN: 포함 & 미검출
- WER : `WER = (S + D + I) / N × 100`
  - S: Substitution, D: Deletion, I: Insertion, N: 레퍼런스 총 단어 수
- Δ WER : `ΔWER = abs(WER_base - WER_test)`

### 테스트 세트 및 환경
- **테스트 영상**: 목표 음향이 포함된 YouTube 웨비나·강의 영상 10편(표 1, whisper_at_report.md 참고)
- **환경**: Debian 11, Python 3.10.17, CUDA 12.4, PyTorch 2.3.1, GPU Tesla T4 (15 GB)  
  (세부 사양은 `TC1_Operating_Environment.md` 참조)
- **시스템 구성**: `TC1_Test_Environment_Diagram.md` 참조

## 3. 예상 결과
| 구분 | 목표 값 |
|------|---------|
| 5개 목표 음향 Accuracy | **100 %** (Precision = Recall = F1 = 1.0) |
| Δ WER | **≤ 0.5 %p** |

## 4. 측정 방법

### 4.1 음향 Accuracy 계산 절차
1. 각 테스트 영상의 레퍼런스 어노테이션(음향 발생 시점, 종류) 준비
2. 모델 출력 태그와 레퍼런스 비교 (±1 초 윈도우 허용)
3. TP·TN·FP·FN 계수 → Accuracy, Precision, Recall, F1 산출
4. 5개 목표 음향 클래스별 결과가 모두 100 %이면 목표 달성

### 4.2 음성 인식 WER 계산 절차
1. 동일 오디오에 대해
   - **WER_base** : Whisper Large-v1 출력 ↔ 레퍼런스 자막 비교
   - **WER_test** : Whisper-AT 출력 ↔ 레퍼런스 자막 비교
2. Δ WER = |WER_base − WER_test|
3. Δ WER ≤ 0.5 %p 이면 음성 인식률 저하 없음

#### Python 예시 코드 (jiwer 사용)
```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])

with open("reference.txt") as f:
    reference = [line.strip() for line in f]
with open("hyp_base.txt") as f:
    hyp_base = [line.strip() for line in f]
with open("hyp_test.txt") as f:
    hyp_test = [line.strip() for line in f]

wer_base = wer(reference, hyp_base, truth_transform=transform, hypothesis_transform=transform)
wer_test = wer(reference, hyp_test, truth_transform=transform, hypothesis_transform=transform)
wer_delta = abs(wer_base - wer_test)

print(f"WER_base : {wer_base*100:.2f}%")
print(f"WER_test : {wer_test*100:.2f}%")
print(f"Δ WER    : {wer_delta*100:.2f} %p")
```

---
본 규격서는 TC1 시험의 평가 항목·목표·기준 및 측정 방법을 정의하며, Whisper-AT 기반 음향·음성 동시 인식 시스템의 성능 검증에 사용된다. 