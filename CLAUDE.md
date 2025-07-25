# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Poetry Environment
This project uses Poetry for dependency management:
- `poetry install` - Install all dependencies
- `poetry shell` - Activate the virtual environment
- `poetry run python script.py` - Run Python scripts in the environment

### Package Installation (Alternative)
For the whisper-at package (located in `package/whisper-at/`):
- `pip install whisper-at` - Install from PyPI
- For Mac/Windows: `pip install numba numpy torch tqdm more-itertools tiktoken==0.3.3` then `pip install --no-deps whisper-at`

### Training Scripts
Model training is done through shell scripts in `src/whisper_at_train/`:
- `./run_as_full_train.sh` - Train on AudioSet full dataset
- `./run_as_sonyc.sh` - Train on SONYC-UST dataset  
- `./run_combined_training.sh` - Combined training approach
- Training uses SLURM job scheduler (modify script paths for local execution)

### Testing and Evaluation
- `python evaluate_pretrained_whisper_at.py` - Evaluate pretrained models
- Sample usage examples in `sample/whisper_transcribe_test_simple.py`
- Jupyter notebook demo: `sample/whisper_at_demo.ipynb`

## Project Architecture

### Core Whisper-AT Package (`package/whisper-at/`)
- **whisper_at/transcribe.py**: Main transcription API with audio tagging support
- **whisper_at/at_post_processing.py**: Audio tagging post-processing utilities
- **whisper_at/model.py**: Modified Whisper model with AT capabilities
- Compatible API with original OpenAI Whisper

### Training Infrastructure (`src/whisper_at_train/`)
- **models.py**: Time and Layer-wise Transformer (TL-TR) model architecture
- **traintest.py**: Main training/testing logic with class weighting support
- **dataloader_feat.py**: Feature-based data loading for pre-extracted representations
- **run.py**: Training script entry point

### Research Code (`src/noise_robust_asr/`)
- **asr_experiments/**: Noise robustness experiments for various ASR models
- **intermediate_feat_extract/**: Feature extraction from different model layers
- **baseline_sound_classification.py**: Sound classification baselines
- **plots/**: Plotting scripts for paper figures

### Key Features
- **Dual Functionality**: Simultaneous speech recognition and audio event tagging
- **Frozen Backbone**: Original Whisper parameters remain unchanged
- **TL-TR Architecture**: Novel Time and Layer-wise Transformer for audio tagging
- **AudioSet Labels**: Outputs 527-class AudioSet event labels
- **Multi-resolution**: Configurable temporal resolution for audio tagging (`at_time_res`)

### Data Processing
- Processed datasets stored in `data/processed_data/`
- AudioSet and SONYC-UST dataset integration
- Class weighting and label mapping utilities in training scripts
- Feature extraction and caching for efficient training

### Gradio Demo
- `app.py`: Web interface for interactive testing
- Supports multiple model sizes and languages
- Real-time audio transcription and tagging

## TC1 Test Environment

### Operating Environment (운영환경)
- **OS**: Debian GNU/Linux 11 (bullseye) - Kernel 5.10.0-35-cloud-amd64
- **Hardware**: Google Cloud Platform GPU VM Instance
  - CPU: Intel Xeon @ 2.0GHz (8 vCPUs, 4 cores with hyperthreading)
  - RAM: 30GB DDR4
  - GPU: NVIDIA Tesla T4 (15GB VRAM)
  - Storage: 296GB root + 2TB NVMe SSD (/mnt/ssd_disk)
- **Software Stack**:
  - Python 3.10.17
  - CUDA 12.4, NVIDIA Driver 550.90.07
  - PyTorch 2.3.1, whisper-at 0.6
  - Key packages: numpy 2.2.5, pandas 2.0.3, scikit-learn 1.6.1

### TC1 Test Specifications
**Objective**: Achieve 100% accuracy in detecting 5 target sounds (applause, laughter, cough, sigh, cheering) in 30dB SNR webinar/lecture videos while maintaining speech recognition performance (ΔWER ≤ 0.5%p).

**Test Dataset**: 10 YouTube webinar/lecture videos containing target sounds:
- Videos include applause, laughter, cough, sigh, and cheering sounds
- Reference annotations with ground truth timestamps (±1 second tolerance)
- 30dB SNR conditions representing real webinar environments

**Evaluation Metrics**:
- Audio Recognition Accuracy: `(TP + TN) / (TP + TN + FP + FN) × 100`
- Word Error Rate Delta: `ΔWER = abs(WER_base - WER_test)`
- Success criteria: 100% accuracy for 5 target sounds + ΔWER ≤ 0.5%p

### Test Procedures
1. **Environment Setup**: Verify GPU VM with required software stack
2. **Baseline WER**: Measure Whisper Large-v1 performance
3. **Test Model Execution**: Run Whisper-AT with SONYC-UST fine-tuning
4. **Audio Accuracy Calculation**: Compare predictions with ground truth annotations
5. **Results Analysis**: Calculate metrics and determine pass/fail criteria

### Model Architecture & Training
**Enhanced Whisper-AT Model**:
- Base: Whisper Large-v1 + Time Layer-wise Transformer (TL-TR)
- Extended Classes: 527 AudioSet + 6 SONYC-UST classes = 533 total classes
- Fine-tuning on combined AudioSet 20k + SONYC-UST dataset
- Performance: mAP 0.4148 overall, mAP 0.4529 for SONYC-UST urban sound classes

**SONYC-UST Extension**: Added 6 new urban sound classes:
- Amplified speech, Hoe ram, Large rotating saw
- Non machinery impact, Pile driver, Small medium rotating saw

## Important Notes

- Audio tagging time resolution (`at_time_res`) must be a multiple of 0.4 seconds
- Default training uses pre-extracted Whisper features to save computation
- Model supports both English-only and multilingual variants
- Compatible with original Whisper API - can be used as drop-in replacement
- TC1 certified for 100% accuracy on 5 target sounds with no speech recognition degradation

## AI Interaction Guidelines

- Always answers in Korean

## Project Memories

### Model Checkpoint Resources
- `package/whisper-at/whisper_at/__init__.py`: Model checkpoint locations in Dropbox paths

### Training Scripts
- `src/whisper_at_train/run_combined_training.sh`: Combined training script for Whisper-AT models

### Evaluation Scripts
- `src/whisper_at_train/evaluate_pretrained_whisper_at.py`: Script for evaluating pretrained Whisper-AT models