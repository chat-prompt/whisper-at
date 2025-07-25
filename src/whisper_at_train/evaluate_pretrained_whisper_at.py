# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import pandas as pd
from types import SimpleNamespace

# whisper-at 프로젝트의 모듈 임포트
try:
    import dataloader_feat as dataloader
    from traintest import validate
    from models import TLTR # whisper-at에서 사용하는 주 모델 아키텍처로 가정
    from whisper.model import ModelDimensions
except ImportError as e:
    print(f"Error importing whisper-at modules: {e}")
    print("Please ensure this script is run within the whisper-at project environment,")
    print("or that the whisper-at source directory is in your PYTHONPATH.")
    exit(1)

def get_model_specific_feat_shape(model_size_str):
    """
    모델 크기 문자열을 기반으로 레이어 수와 표현 차원을 반환합니다.
    """
    n_rep_dim_dict = {
        'tiny.en': 384, 'tiny': 384, 'base.en': 512, 'base': 512,
        'small.en': 768, 'small': 768, 'medium.en': 1024, 'medium': 1024,
        'large-v1': 1280, 'large-v2': 1280, 'large-v3': 1280,
        'wav2vec2-large-robust-ft-swbd-300h': 1024,
        'hubert-xlarge-ls960-ft': 1280
    }
    n_layer_dict = {
        'tiny.en': 4, 'tiny': 4, 'base.en': 6, 'base': 6,
        'small.en': 12, 'small': 12, 'medium.en': 24, 'medium': 24,
        'large-v1': 32, 'large-v2': 32, 'large-v3': 32,
        'wav2vec2-large-robust-ft-swbd-300h': 24,
        'hubert-xlarge-ls960-ft': 48
    }
    if model_size_str not in n_layer_dict or model_size_str not in n_rep_dim_dict:
        raise ValueError(f"Unsupported model_size for get_model_specific_feat_shape: {model_size_str}")
    return n_layer_dict[model_size_str], n_rep_dim_dict[model_size_str]

def load_label_mapping(label_csv_path):
    """레이블 CSV 파일에서 인덱스-클래스명 매핑을 로드합니다."""
    try:
        # audioset_label.csv는 헤더가 없고, index, mid, display_name 순서로 가정
        df = pd.read_csv(label_csv_path, header=None, names=['index', 'mid', 'display_name'])
        if not (isinstance(df['index'].iloc[0], (int, np.integer)) and isinstance(df['display_name'].iloc[0], str)):
            print("Warning: CSV format might not match expected 'index, mid, display_name' without header. Trying with header.")
            df = pd.read_csv(label_csv_path) # 헤더가 있는 경우 다시 시도
            if 'index' not in df.columns or 'display_name' not in df.columns:
                 raise ValueError("Label CSV must contain 'index' and 'display_name' columns or be parsable as headerless 'index,mid,display_name'.")
        return pd.Series(df.display_name.values, index=df['index']).to_dict()
    except FileNotFoundError:
        print(f"Error: Label CSV file not found at {label_csv_path}")
        raise
    except Exception as e:
        print(f"Error loading or processing label CSV file: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained Whisper-AT model on AudioSet test set.")
    parser.add_argument("--model_architecture", type=str, required=True,
                        help="Name of the Whisper-AT model architecture (e.g., 'whisper-high-lw_tr_1_8').")
    parser.add_argument("--model_size", type=str, required=True,
                        choices=['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3', 'tiny.en', 'base.en', 'small.en', 'medium.en'],
                        help="Size of the Whisper base model.")
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path to the pre-trained Whisper-AT model checkpoint.")
    parser.add_argument("--test_data_json", type=str, required=True,
                        help="Path to the AudioSet test dataset JSON file.")
    parser.add_argument("--label_csv", type=str, required=True,
                        help="Path to the label mapping CSV file.")
    parser.add_argument("--n_class", type=int, default=527, help="Number of classes.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use.")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"Instantiating model: {args.model_architecture} with size {args.model_size}")
    try:
        n_layer, rep_dim = get_model_specific_feat_shape(args.model_size)
        mode_parts = args.model_architecture.split('-') # "whisper-high-lw_tr_1_8"
        
        # 모드 추출 로직 개선 (예: "lw_tr_1_8" 또는 "tr_1_8" 등)
        # whisper-at 코드의 모델 이름 규칙을 따라야 함
        # run.py: mode = args.model.split('-')[-1]
        # 이 방식은 'whisper-high-lw_tr_1_8' -> 'lw_tr_1_8'
        # 'whisper-high-tr_1_8' -> 'tr_1_8'
        mode = mode_parts[-1]
        print(f"Using mode: {mode} for TLTR model.")
        model = TLTR(label_dim=args.n_class, n_layer=n_layer, rep_dim=rep_dim, mode=mode)
    except Exception as e:
        print(f"Error instantiating model architecture: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print(f"Loading pre-trained weights from: {args.pretrained_model_path}")
    try:
        checkpoint = torch.load(args.pretrained_model_path, map_location=device)
        state_dict_to_load = None
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict_to_load = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict_to_load = checkpoint['state_dict']
            elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                 state_dict_to_load = checkpoint['model']
            else: # 딕셔너리이지만 주요 키가 없는 경우, 딕셔너리 자체가 state_dict일 수 있음
                print("Checkpoint is a dictionary but does not contain standard state_dict keys. Attempting to load the dictionary itself as state_dict.")
                state_dict_to_load = checkpoint 
        else: # 딕셔너리가 아니면 state_dict 자체로 간주
            state_dict_to_load = checkpoint

        if state_dict_to_load is None:
            raise ValueError("Could not extract state_dict from the checkpoint file.")

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict_to_load.items():
            name = k
            if name.startswith('module.'): # DataParallel 접두사 제거
                name = name[7:]
            if name.startswith('at_model.'): # 'at_model.' 접두사 추가 제거
                name = name[9:]
            new_state_dict[name] = v
        
        load_result = model.load_state_dict(new_state_dict, strict=False) # strict=False로 하여 일부 키 불일치 허용
        if load_result.missing_keys:
            print(f"Warning: Missing keys when loading state_dict: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Warning: Unexpected keys when loading state_dict: {load_result.unexpected_keys}")

        model.to(device)
        model.eval()
        print("Pre-trained weights loaded successfully.")
    except Exception as e:
        print(f"Error loading pre-trained weights: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print(f"Creating DataLoader for: {args.test_data_json}")
    try:
        eval_tar_path = '/mnt/ssd_disk/datasets/audioset_features_eval/features/whisper_large-v1'
        eval_audio_conf = {'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'audioset', 'label_smooth': 0.0, 'tar_path': eval_tar_path}
        # dataloader.AudiosetDataset 호출 시 n_class 인자 제거 (데이터로더가 label_csv로 결정)
        test_dataset = dataloader.AudiosetDataset(args.test_data_json, label_csv=args.label_csv, audio_conf=eval_audio_conf)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print(f"DataLoader created with {len(test_dataset)} samples.")
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("Starting evaluation...")
    try:
        validate_args = SimpleNamespace()
        validate_args.loss_fn = torch.nn.BCEWithLogitsLoss()
        validate_args.metrics = 'mAP'
        # validate_args.n_class = args.n_class # validate 함수는 내부적으로 target shape에서 클래스 수를 얻음

        if not isinstance(model, torch.nn.DataParallel) and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for evaluation via DataParallel.")
            model = torch.nn.DataParallel(model)
        
        # validate 함수의 반환 값을 먼저 받고 None인지 확인
        validation_output = validate(model, test_loader, validate_args)

        if validation_output is None:
            print("Error: The 'validate' function returned None. Cannot proceed with metrics calculation.")
            stats = [] # 빈 리스트로 초기화하여 이후 코드에서 오류 방지
        else:
            stats, _ = validation_output # None이 아니면 언패킹

        if not stats: # stats가 비어있거나 None일 경우
             print("No valid stats returned from validation.")
        else:
            valid_aps = [stat['AP'] for stat in stats if isinstance(stat, dict) and 'AP' in stat]
            if valid_aps:
                 mAP = np.mean(valid_aps)
                 print(f"  Calculated mAP: {mAP:.6f}")
            else:
                 print("  Could not calculate mAP from stats.")

            valid_aucs = [stat['auc'] for stat in stats if isinstance(stat, dict) and 'auc' in stat]
            if valid_aucs:
                mAUC = np.mean(valid_aucs)
                print(f"  Calculated mAUC: {mAUC:.6f}")
            else:
                 print("  mAUC not found in stats.")
            
            print("\nCompare these results with the reported results in the whisper-at paper for the corresponding model and dataset.")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\nEvaluation finished.")

if __name__ == "__main__":
    main()
