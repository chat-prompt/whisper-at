# -*- coding: utf-8 -*-
# @Time    : 2025-05-08
# @Author  : Gemini AI
# @File    : extract_hf_audioset.py

import os
import json
import argparse
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import skimage.measure
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

# 공식 whisper 패키지 임포트
try:
    import whisper
except ImportError:
    print("Please install OpenAI Whisper: pip install openai-whisper")
    whisper = None

# whisper-at의 모델 로딩을 위한 클래스
try:
    from whisper.model import Whisper, ModelDimensions
except ImportError:
    print("Please ensure the 'whisper.model' module from your 'whisper-at' environment is in the PYTHONPATH for ModelDimensions.")
    if whisper is not None:
        Whisper = whisper.Whisper
    else:
        Whisper = None
    ModelDimensions = None

# Predefined model dimensions (이전과 동일)
_MODELS = {
    "tiny": ModelDimensions(n_mels=80, n_vocab=51865, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6, n_audio_layer=4, n_text_ctx=448, n_text_state=384, n_text_head=6, n_text_layer=4),
    "base": ModelDimensions(n_mels=80, n_vocab=51865, n_audio_ctx=1500, n_audio_state=512, n_audio_head=8, n_audio_layer=6, n_text_ctx=448, n_text_state=512, n_text_head=8, n_text_layer=6),
    "small": ModelDimensions(n_mels=80, n_vocab=51865, n_audio_ctx=1500, n_audio_state=768, n_audio_head=12, n_audio_layer=12, n_text_ctx=448, n_text_state=768, n_text_head=12, n_text_layer=12),
    "medium": ModelDimensions(n_mels=80, n_vocab=51865, n_audio_ctx=1500, n_audio_state=1024, n_audio_head=16, n_audio_layer=24, n_text_ctx=448, n_text_state=1024, n_text_head=16, n_text_layer=24),
    "large-v1": ModelDimensions(n_mels=80, n_vocab=51865, n_audio_ctx=1500, n_audio_state=1280, n_audio_head=20, n_audio_layer=32, n_text_ctx=448, n_text_state=1280, n_text_head=20, n_text_layer=32),
    "large-v2": ModelDimensions(n_mels=80, n_vocab=51865, n_audio_ctx=1500, n_audio_state=1280, n_audio_head=20, n_audio_layer=32, n_text_ctx=448, n_text_state=1280, n_text_head=20, n_text_layer=32),
    "large-v3": ModelDimensions(n_mels=128, n_vocab=51865, n_audio_ctx=1500, n_audio_state=1280, n_audio_head=20, n_audio_layer=32, n_text_ctx=448, n_text_state=1280, n_text_head=20, n_text_layer=32),
    "tiny.en": ModelDimensions(n_mels=80, n_vocab=51864, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6, n_audio_layer=4, n_text_ctx=448, n_text_state=384, n_text_head=6, n_text_layer=4),
    "base.en": ModelDimensions(n_mels=80, n_vocab=51864, n_audio_ctx=1500, n_audio_state=512, n_audio_head=8, n_audio_layer=6, n_text_ctx=448, n_text_state=512, n_text_head=8, n_text_layer=6),
    "small.en": ModelDimensions(n_mels=80, n_vocab=51864, n_audio_ctx=1500, n_audio_state=768, n_audio_head=12, n_audio_layer=12, n_text_ctx=448, n_text_state=768, n_text_head=12, n_text_layer=12),
    "medium.en": ModelDimensions(n_mels=80, n_vocab=51864, n_audio_ctx=1500, n_audio_state=1024, n_audio_head=16, n_audio_layer=24, n_text_ctx=448, n_text_state=1024, n_text_head=16, n_text_layer=24),
}

def get_model_dimensions(model_name: str):
    if ModelDimensions is None:
        raise ImportError("ModelDimensions could not be imported from whisper.model. Check your whisper-at environment.")
    if model_name in _MODELS:
        return _MODELS[model_name]
    else:
        if model_name.endswith(".en") and model_name[:-3] in _MODELS:
            return _MODELS[model_name[:-3]]
    raise ValueError(f"Model dimensions not found for {model_name}. Known models: {list(_MODELS.keys())}")


def extract_features_from_hf_dataset(
    hf_dataset_name,
    hf_dataset_split,
    max_samples,
    whisper_model_instance,
    output_feature_dir,
    output_json_path,
    model_size_name,
    device
):
    if whisper is None:
        print("OpenAI Whisper module not loaded. Exiting.")
        return

    config_name = None
    actual_hf_split_name = None
    if hf_dataset_split == "balanced_train":
        config_name = "balanced"
        actual_hf_split_name = "train"
    elif hf_dataset_split == "unbalanced_train":
        config_name = "unbalanced"
        actual_hf_split_name = "train"
    elif hf_dataset_split == "eval":
        config_name = "balanced"
        actual_hf_split_name = "test"
    else:
        raise ValueError(
            f"Invalid --hf_dataset_split value: '{hf_dataset_split}'. "
            f"Please use 'balanced_train', 'unbalanced_train', or 'eval'."
        )

    print(f"Loading Hugging Face dataset: {hf_dataset_name}, Config: '{config_name}', Split: '{actual_hf_split_name}'")
    try:
        dataset = load_dataset(hf_dataset_name, name=config_name, split=actual_hf_split_name, streaming=False, trust_remote_code=True)
        # 오디오를 미리 로드하고 디코딩하도록 요청 (16kHz로 캐스팅)
        # 이 과정에서 문제가 있는 파일은 미리 걸러지지 않을 수 있음 (오류는 반복 접근 시 발생)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if not os.path.exists(output_feature_dir):
        os.makedirs(output_feature_dir)
        print(f"Created output feature directory: {output_feature_dir}")

    all_metadata = []
    
    # 처리할 샘플 수 결정
    num_samples_in_dataset = len(dataset)
    num_samples_to_process = num_samples_in_dataset
    if max_samples is not None and max_samples > 0:
        num_samples_to_process = min(max_samples, num_samples_in_dataset)

    progress_bar_desc = f"Extracting features for {hf_dataset_split} ({model_size_name})"
    
    problematic_item_indices = []

    for i in tqdm(range(num_samples_to_process), desc=progress_bar_desc, unit="sample"):
        try:
            # dataset[i] 접근 시 오디오 디코딩이 발생하며, 여기서 LibsndfileError가 날 수 있음
            item = dataset[i]
            
            audio_data_dict = item['audio']
            if audio_data_dict is None or audio_data_dict['array'] is None:
                tqdm.write(f"Skipping item at index {i} due to missing audio data after access.")
                problematic_item_indices.append({"index": i, "id": item.get('video_id_str', 'N/A'), "error": "Missing audio data"})
                continue
            
            audio_array = audio_data_dict['array']

            youtube_id = item.get('video_id', item.get('audio_id', f"unknown_id_{i}"))
            sanitized_youtube_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in youtube_id)
            base_filename = f"{sanitized_youtube_id}_{i}"
            feature_filename = f"{base_filename}.npz"
            feature_filepath = os.path.join(output_feature_dir, feature_filename)

            if os.path.exists(feature_filepath):
                # tqdm.write(f"Features for {youtube_id} (item {i}) already exist: {feature_filepath}. Skipping extraction.")
                pass # 이미 처리된 파일은 그냥 넘어감
            else:
                target_length = 30 * whisper.audio.SAMPLE_RATE
                padded_audio = np.zeros(target_length, dtype=np.float32)
                
                current_audio_len = len(audio_array)
                if current_audio_len > target_length:
                    padded_audio = audio_array[:target_length]
                else:
                    padded_audio[:current_audio_len] = audio_array
                
                mel = whisper.log_mel_spectrogram(padded_audio, n_mels=whisper_model_instance.dims.n_mels).to(device)
                
                with torch.no_grad():
                    x = mel.unsqueeze(0)
                    x = whisper_model_instance.encoder.conv1(x)
                    x = F.gelu(x)
                    x = whisper_model_instance.encoder.conv2(x)
                    x = F.gelu(x)
                    x = x.permute(0, 2, 1)
                    x = (x + whisper_model_instance.encoder.positional_embedding[:x.size(1)]).to(x.dtype)
                    
                    layer_features = []
                    for block_idx, block in enumerate(whisper_model_instance.encoder.blocks):
                        x = block(x)
                        layer_output = x.clone()
                        num_frames_10s = whisper_model_instance.dims.n_audio_ctx // 3
                        first_10s_output = layer_output[:, :num_frames_10s, :]
                        pooled = F.avg_pool1d(
                            first_10s_output.permute(0, 2, 1),
                            kernel_size=20,
                            stride=20
                        ).permute(0, 2, 1)
                        layer_features.append(pooled)
                
                stacked_features = torch.stack(layer_features, dim=0)
                features_np = stacked_features.cpu().numpy()[:, 0, :, :]
                np.savez_compressed(feature_filepath, features_np)

            raw_labels = item.get('labels', [])
            metadata_entry = {
                "hf_item_index": i, # 현재 처리 중인 인덱스
                "youtube_id": youtube_id,
                "audio_hf_path": audio_data_dict.get('path', 'N/A_in_memory_array'),
                "feature_path": os.path.relpath(feature_filepath, os.path.dirname(output_json_path)),
                "labels": raw_labels
            }
            all_metadata.append(metadata_entry)

        except sf.LibsndfileError as sf_error:
            youtube_id_str = "N/A"
            try: # 오류 발생 시에도 ID를 가져오려는 시도 (item 객체가 부분적으로라도 생성되었을 수 있음)
                item_for_id = dataset[i] # 다시 접근 시도 (이미 오류난 파일일 수 있음)
                youtube_id_str = item_for_id.get('video_id_str', f"unknown_id_at_idx_{i}")
            except: # ID 가져오기 실패 시
                pass
            tqdm.write(f"SoundfileError processing item at index {i} (ID: {youtube_id_str}): {sf_error}. Skipping this item.")
            problematic_item_indices.append({"index": i, "id": youtube_id_str, "error": str(sf_error)})
            continue # 다음 아이템으로 넘어감
        except Exception as e:
            youtube_id_str = "N/A"
            try:
                item_for_id = dataset[i]
                youtube_id_str = item_for_id.get('video_id_str', f"unknown_id_at_idx_{i}")
            except:
                pass
            tqdm.write(f"Generic error processing item at index {i} (ID: {youtube_id_str}): {e}")
            import traceback
            tqdm.write(traceback.format_exc())
            problematic_item_indices.append({"index": i, "id": youtube_id_str, "error": str(e)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue # 다음 아이템으로 넘어감

    final_processed_count = len(all_metadata)
    output_data_json = {"data": all_metadata}
    with open(output_json_path, 'w') as f:
        json.dump(output_data_json, f, indent=2)
    print(f"\nSuccessfully processed {final_processed_count} out of {num_samples_to_process} attempted samples.")
    print(f"Feature metadata saved to: {output_json_path}")

    if problematic_item_indices:
        print(f"\nEncountered errors with {len(problematic_item_indices)} items. See below for details or log file.")
        problematic_log_path = os.path.join(os.path.dirname(output_json_path), "problematic_audio_files.json")
        with open(problematic_log_path, 'w') as f:
            json.dump(problematic_item_indices, f, indent=2)
        print(f"Details of problematic items saved to: {problematic_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract AudioSet features from Hugging Face using Whisper.")
    # 인자 파서는 이전과 동일하게 유지
    parser.add_argument("--hf_dataset_name", type=str, default="agkphysics/AudioSet", help="Name of the Hugging Face dataset.")
    parser.add_argument("--hf_dataset_split", type=str, default="balanced_train", choices=["balanced_train", "unbalanced_train", "eval"], help="User-friendly dataset split.")
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum number of samples to process. Set to 0 or None for all from the split.") # 0이면 전체
    parser.add_argument("--whisper_model_size", type=str, default="medium", choices=['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3', 'tiny.en', 'base.en', 'small.en', 'medium.en'], help="Size of the Whisper model.")
    parser.add_argument("--whisper_checkpoint_path", type=str, required=True, help="Path to the Whisper model checkpoint (.pt or .pth file).")
    parser.add_argument("--output_dir", type=str, default="./audioset_extracted_features", help="Base directory to save features and metadata.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for Torch (cuda or cpu).")

    args = parser.parse_args()

    if args.max_samples == 0:
        args.max_samples = None # 0이면 None으로 설정하여 전체 데이터셋을 의미하도록 함

    if ModelDimensions is None :
         raise ImportError("ModelDimensions could not be imported from whisper.model. Check your whisper-at environment.")

    print(f"Loading Whisper model: {args.whisper_model_size} from {args.whisper_checkpoint_path}")
    device = torch.device(args.device)
    dims_obj = None
    model_state_dict = None

    try:
        checkpoint = torch.load(args.whisper_checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            if "dims" in checkpoint:
                dims_data = checkpoint["dims"]
                dims_obj = ModelDimensions(**dims_data) if isinstance(dims_data, dict) else dims_data
            else:
                dims_obj = get_model_dimensions(args.whisper_model_size)
            model_state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            dims_obj = get_model_dimensions(args.whisper_model_size)
            model_state_dict = checkpoint

        if dims_obj is None or model_state_dict is None:
            raise ValueError("Failed to determine model dimensions or state_dict.")

        if 'Whisper' not in globals() or Whisper is None or not hasattr(Whisper, '__init__'):
            raise ImportError("The Whisper class for model definition (from whisper.model) is not correctly imported or defined.")

        whisper_model_instance = Whisper(dims_obj)
        whisper_model_instance.load_state_dict(model_state_dict, strict=False)
        whisper_model_instance.eval()
        whisper_model_instance.to(device)
        print(f"Whisper model {args.whisper_model_size} loaded successfully to {args.device}.")

    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    model_specific_feature_subdir = f"whisper_{args.whisper_model_size}"
    output_feature_storage_dir = os.path.join(args.output_dir, "features", model_specific_feature_subdir)
    output_metadata_dir = os.path.join(args.output_dir, "metadata")
    
    if not os.path.exists(output_metadata_dir):
        os.makedirs(output_metadata_dir)

    max_samples_str = str(args.max_samples) if args.max_samples is not None else 'all'
    output_json_filename = f"audioset_{args.hf_dataset_split.replace('_','-')}_{args.whisper_model_size}_{max_samples_str}_features.json"
    output_json_full_path = os.path.join(output_metadata_dir, output_json_filename)

    extract_features_from_hf_dataset(
        hf_dataset_name=args.hf_dataset_name,
        hf_dataset_split=args.hf_dataset_split,
        max_samples=args.max_samples,
        whisper_model_instance=whisper_model_instance,
        output_feature_dir=output_feature_storage_dir,
        output_json_path=output_json_full_path,
        model_size_name=args.whisper_model_size,
        device=device
    )

    print("Feature extraction process complete.")
