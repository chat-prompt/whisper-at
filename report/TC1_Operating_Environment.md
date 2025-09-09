# TC1(음성 외 음향 인식률) 운영환경

## 소프트웨어 정보

| 항목 | 내용 |
|------|------|
| **운영체제** | Debian GNU/Linux 11 (bullseye) - Kernel 5.10.0-35-cloud-amd64 |
| **특이사항(제품구동요구사항)** | Python 3.10.17, CUDA Version 12.4, Git 2.30.2, NVIDIA Driver 550.90.07 |
| **실행 패키지 목록** | torch 2.3.1, whisper-at 0.6, numpy 2.2.5, pandas 2.0.3, scikit-learn 1.6.1, torch-xla 2.4.0, torchvision 0.19.0+cu124 |

## 하드웨어 정보

| 항목 | 내용 |
|------|------|
| **하드웨어 사양** | **CPU**: Intel(R) Xeon(R) CPU @ 2.00GHz, 8 vCPUs (4코어, 하이퍼스레딩)<br/>**RAM**: 30GB DDR4<br/>**스토리지**: 300GB (루트 파티션) + 2TB SSD (GCP Persistent Disk, SCSI 인터페이스, /mnt/ssd_disk)<br/>**GPU**: NVIDIA Tesla T4 (15GB VRAM) |
| **네트워크 환경** | TCP/IP (고속 인터넷 연결) |
| **기타 환경** | Google Cloud Platform (GCP) GPU VM Instance |

## 기타사항

- 신청기업(지피터스)에서 필요한 HW 및 SW를 제공함
- 신청기업(지피터스)에서 시험환경구성, 제품설치 및 기술지원을 지원함

---

## 상세 시스템 정보

### CPU 정보
- **Architecture**: x86_64
- **CPU(s)**: 8
- **Thread(s) per core**: 2
- **Core(s) per socket**: 4
- **Socket(s)**: 1
- **Model name**: Intel(R) Xeon(R) CPU @ 2.00GHz
- **CPU MHz**: 2000.140
- **L1d cache**: 128 KiB
- **L1i cache**: 128 KiB
- **L2 cache**: 4 MiB
- **L3 cache**: 38.5 MiB

### GPU 정보
- **GPU 모델**: NVIDIA Tesla T4
- **GPU 메모리**: 15,360 MiB (15GB)
- **CUDA 버전**: 12.4
- **Driver 버전**: 550.90.07

### 스토리지 정보
- **루트 파티션 (/dev/sda1)**: 296GB (사용: 176GB, 가용: 108GB)
- **데이터 디스크 (/dev/sdb1)**: 2.0TB (사용: 530GB, 가용: 1.4TB, 마운트: /mnt/ssd_disk)

### 메모리 정보
- **총 메모리**: 30GB
- **사용 중**: 1.2GB
- **가용**: 27GB
- **버퍼/캐시**: 2.6GB 