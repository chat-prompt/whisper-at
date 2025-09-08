# TC1(음성 외 음향 인식률) 모델 및 데이터 요약

## 1. 모델
- **모델명**: Whisper-AT Large-v1 파인튜닝(533 클래스)
- **기능**: 웨비나/강의 오디오에서 음성 인식(ASR)과 음향 이벤트 태깅 동시 수행
- **유형**: 멀티태스크 오디오 인식(ASR + Audio Tagging)
- **구조**
  - 기반: Whisper Large-v1 인코더
  - 태깅 헤드: 시간적 풀링(Temporal Pooling) → 선형 투사(Linear Projection) → 추가 트랜스포머 → MLP 분류기(527→533 클래스 확장)
  - 목표 음향: 박수, 웃음, 기침, 한숨, 환호(5종) 포함 총 533 클래스
  - 전처리: 16 kHz 리샘플링 및 정규화
- **학습**
  - 훈련 데이터셋: AudioSet 20k + SONYC-UST 결합, 신규 6개 도시소음 클래스 추가
  - 검증 데이터셋: AudioSet/SONYC-UST 검증 세트(결합 평가, mAP 기준)
- **입출력**
  - 입력: 16 kHz mono WAV(유튜브 영상에서 추출)
  - 출력: 
    - 음성 인식 텍스트(자막)
    - 음향 태그 JSON(533 클래스 확률/태그, 특히 5개 목표 음향의 시점 태깅)

---

## 2. 데이터
- **데이터명**: TC1 평가용 웨비나/강의 영상 10편 테스트셋
- **형식**: 
  - 오디오: 16 kHz WAV(yt-dlp+ffmpeg로 추출)
  - 자막: 정제·정규화된 레퍼런스 텍스트(.txt)
  - 어노테이션: 목표 음향 타임스탬프 레이블(.json, ±1초 판정)
- **테스트 샘플 수**: 164개 (5가지 목표 음향 총합)
  - 박수 소리: 45개 (9개 파일)
  - 웃음 소리: 107개 (5개 파일) 
  - 기침 소리: 6개 (1개 파일)
  - 한숨 소리: 3개 (1개 파일)
  - 환호하는 소리: 3개 (1개 파일)
- **샘플**
  - https://www.youtube.com/watch?v=8S0FDjFBj8o — 박수(3), 웃음(24)
  - https://www.youtube.com/watch?v=9kxL9Cf46VM — 박수(10), 웃음(21)
  - https://www.youtube.com/watch?v=AzIhz0kE8SA — 박수(7)
  - https://www.youtube.com/watch?v=Fkd9TWUtFm0 — 박수(1), 웃음(6)
  - https://www.youtube.com/watch?v=QFqokhs47l0 — 박수(5), 웃음(51)
  - https://www.youtube.com/watch?v=a3dmC2nB-vE — 기침(6)
  - https://www.youtube.com/watch?v=azRl1dI-Cts — 박수(6)
  - https://www.youtube.com/watch?v=hFcQpNr_KA4 — 박수(3), 한숨(3)
  - https://www.youtube.com/watch?v=vVsXO9brK7M — 박수(6), 웃음(5)
  - https://www.youtube.com/watch?v=vvi1hCoFAgo — 박수(4), 환호(3)

---

### 참고 문서
- `report/TC1_Operating_Environment.md`
- `report/TC1_Test_Environment_Diagram.md`
- `report/TC1_Test_Procedure.md`
- `report/TC1_Test_Specification.md`
- `report/whisper_at_report.md`
