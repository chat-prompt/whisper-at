# TC1(음성 외 음향 인식률) 시험 절차서

## 1. 시험 항목명
- **음성 외 음향 인식률**

## 2. 시험 목표 및 기준
- <시험목표>
  - 30 dB SNR 조건의 실제 웨비나·강의 영상 10편에서 5가지 목표 음향(박수, 웃음, 기침, 한숨, 환호)을 **100 % 정확도**로 검출·분류하면서, 기존 Whisper Large-v1 대비 **ΔWER ≤ 0.5 %p** 를 달성하는지 확인
- <시험기준>
  1. **음향 Accuracy** : `Accuracy = (TP + TN) / (TP + TN + FP + FN) × 100`
  2. **ΔWER** : `ΔWER = abs(WER_base − WER_test)`
     - WER_base : Whisper Large-v1 결과 ↔ 레퍼런스 자막
     - WER_test : Whisper-AT 결과 ↔ 레퍼런스 자막
  3. 합격 기준  
     - 5개 목표 음향 Accuracy = **100 %**  
     - ΔWER ≤ **0.5 %p**

## 3. 사전 조건
- 운영체제 : Debian GNU/Linux 11 (bullseye) ‑ Kernel 5.10.0-35-cloud-amd64
- Python 3.10.17, Poetry 2.1.2 설치
- GPU / CUDA | NVIDIA Tesla T4 (15 GB VRAM) / CUDA 12.4
- 주요 패키지
  - torch 2.3.1
  - whisper-at 0.6
  - numpy 2.2.5
  - pandas 2.0.3
  - scikit-learn 1.6.1
- 기타
  - Git 2.30.2, ffmpeg 설치, 16 kHz wav 변환용 스크립트 구비

## 4. 시료
- **테스트 영상 10편**
  - https://www.youtube.com/watch?v=8S0FDjFBj8o : 박수, 웃음
  - https://www.youtube.com/watch?v=hFcQpNr_KA4 : 박수, 한숨
  - https://www.youtube.com/watch?v=a3dmC2nB-vE : 기침
  - https://www.youtube.com/watch?v=AzIhz0kE8SA : 박수
  - https://www.youtube.com/watch?v=azRl1dI-Cts : 박수
  - https://www.youtube.com/watch?v=vVsXO9brK7M : 박수, 웃음
  - https://www.youtube.com/watch?v=QFqokhs47l0 : 박수, 웃음
  - https://www.youtube.com/watch?v=vvi1hCoFAgo : 박수, 환호
  - https://www.youtube.com/watch?v=9kxL9Cf46VM : 박수, 웃음
  - https://www.youtube.com/watch?v=Fkd9TWUtFm0 : 박수, 웃음
- 각 영상의 레퍼런스 자막(정제·정규화 완료) 및 목표 음향 타임스탬프 어노테이션 파일(.json) 보유

## 5. 반복시험 횟수
- **1회** (10편 전체 세트를 1차례 수행)

## 6. 시험 절차
1. **환경 준비**  
   1) GPU VM(GCP) 인스턴스에 SSH 접속  
   2) 다음 필수 SW 패키지가 설치·정상 동작하는지 확인  
      - Python 3.10.17  
      - CUDA Toolkit 12.4 & NVIDIA Driver 550.90.07  
      - PyTorch 2.3.1 (+ torchvision 0.19.0+cu124, torch-xla 2.4.0)  
      - whisper-at 0.6  
      - numpy 2.2.5, pandas 2.0.3, scikit-learn 1.6.1  
      - Git 2.30.2, ffmpeg, yt-dlp  
      - 기타: tqdm, jiwer, soundfile (스크립트 의존 모듈)
2. **Baseline WER 측정**  
   ```bash
   python run_whisper.py \
       --model large-v1 \
       --input_dir ./wav \
       --output_dir ./baseline_transcript
   python calc_wer.py --ref ./refs.txt --hyp ./baseline_transcript.txt --out wer_base.txt
   ```
3. **시험 모델 실행 (Whisper-AT)**  
   ```bash
   python run_whisper_at.py \
       --checkpoint whisper_at_large_finetuned.pth \
       --input_dir ./wav \
       --output_dir ./test_transcript \
       --output_tags ./test_tags.json
   python calc_wer.py --ref ./refs.txt --hyp ./test_transcript.txt --out wer_test.txt
   ```
4. **음향 Accuracy 계산**  
   ```bash
   python eval_audio_tags.py \
       --ref_tags ./gt_tags.json \
       --pred_tags ./test_tags.json \
       --window_sec 1.0 \
       --out accuracy.json
   ```
5. **결과 집계**  
   - `accuracy.json` 에서 5개 목표 음향 Precision·Recall·F1 확인  
   - `wer_base.txt`, `wer_test.txt`에서 ΔWER 계산  
   - 합격 여부 판정 (Accuracy = 100 %, ΔWER ≤ 0.5 %p)
6. **보고서 작성**  
   - 시험 로그, 스크립트, 결과 파일을 `/results/tc1_yyyymmdd` 폴더에 보관  
   - `TC1_Test_Result.md` 양식에 데이터 입력 후 담당자 검토

## 7. 예상 결과
| 지표 | 목표 값 |
|------|---------|
| 5개 목표 음향 Accuracy | 100 % |
| ΔWER | ≤ 0.5 %p |

## 8. 예외 사항
- YouTube 영상 접근 제한(저작권·지역 제한 등)
- ffmpeg 또는 yt-dlp 오류로 인한 오디오 추출 실패
- GPU 메모리 부족(다중 프로세스 실행 시)
- Whisper-AT 체크포인트 손상 또는 불일치
- 레퍼런스 자막과 실제 음원 간 싱크 불일치
- 네트워크 장애로 모델 가중치 다운로드 실패 