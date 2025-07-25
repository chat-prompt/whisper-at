# -*- coding: utf-8 -*-
import os
import argparse
import shutil
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm # tqdm은 여러 파일 처리 시 유용하지만, 단일 파일에는 불필요할 수 있음

def extract_single_audio_from_hf(
    hf_dataset_name,
    hf_config_name, # 데이터셋 설정 이름 (예: 'balanced', 'unbalanced')
    hf_split_name,  # 데이터셋 스플릿 이름 (예: 'train', 'test')
    item_index,
    output_dir
):
    """
    Hugging Face 데이터셋에서 특정 인덱스의 오디오 파일을 추출하여 저장합니다.

    Args:
        hf_dataset_name (str): Hugging Face 데이터셋 이름 (예: "agkphysics/AudioSet").
        hf_config_name (str): 데이터셋 설정 이름 (예: "balanced").
        hf_split_name (str): 데이터셋 스플릿 이름 (예: "train").
        item_index (int): 추출할 아이템의 인덱스 (hf_item_index).
        output_dir (str): 추출된 오디오 파일을 저장할 디렉토리.
    """
    print(f"Loading Hugging Face dataset: {hf_dataset_name}, Config: '{hf_config_name}', Split: '{hf_split_name}'")
    try:
        # 데이터셋 로드 시 오디오를 16kHz로 캐스팅 (필요에 따라 원본 유지 가능)
        # 원본 FLAC을 그대로 가져오려면 Audio() 캐스팅을 하지 않거나,
        # 또는 dataset[item_index]['audio']['path']를 활용해야 함.
        # 여기서는 Audio()로 캐스팅하여 array를 얻고, 이를 FLAC으로 저장하는 방식을 기본으로 함.
        # 만약 원본 FLAC 경로를 직접 복사하고 싶다면, Audio() 캐스팅 없이 로드하고
        # item['audio']가 파일 경로 문자열인지 확인해야 함 (데이터셋마다 다름)
        dataset = load_dataset(hf_dataset_name, name=hf_config_name, split=hf_split_name, streaming=False, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        print(f"Accessing item at index: {item_index}")
        if item_index < 0 or item_index >= len(dataset):
            print(f"Error: Item index {item_index} is out of bounds for dataset with {len(dataset)} samples.")
            return
            
        item = dataset[item_index]
        
        # audio_data_dict는 {'path': ..., 'array': ..., 'sampling_rate': ...} 형태
        # 'agkphysics/AudioSet'의 경우, 'audio' 필드는 Audio() feature로 디코딩됨
        audio_data_dict = dataset.cast_column("audio", Audio(decode=True))[item_index]['audio']


        if audio_data_dict is None:
            print(f"Error: No audio data found for item at index {item_index}.")
            return
            
        audio_array = audio_data_dict.get('array')
        sampling_rate = audio_data_dict.get('sampling_rate')
        original_path_in_cache = audio_data_dict.get('path') # 캐시 내 원본 파일 경로 (있을 수도, 없을 수도)

        if audio_array is None or sampling_rate is None:
            print(f"Error: Audio array or sampling rate is missing for item at index {item_index}.")
            # 만약 original_path_in_cache가 있고, FLAC 파일이라면 이를 직접 복사 시도 가능
            if original_path_in_cache and original_path_in_cache.lower().endswith(".flac"):
                print(f"Attempting to copy directly from cache: {original_path_in_cache}")
            else:
                return

        # 파일명 결정 (youtube_id 사용)
        # agkphysics/AudioSet은 'video_id' 필드에 유튜브 ID 저장
        youtube_id = item.get('video_id', f"unknown_id_idx{item_index}")
        sanitized_youtube_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in youtube_id)
        
        output_filename = f"{sanitized_youtube_id}.flac"
        output_filepath = os.path.join(output_dir, output_filename)

        print(f"Attempting to save audio for ID '{youtube_id}' (index {item_index}) to '{output_filepath}'")

        # 방법 1: 캐시된 원본 FLAC 파일 직접 복사 (더 선호됨, 원본 보존)
        # 'path'가 실제 FLAC 파일을 가리키는지 확인 필요
        if original_path_in_cache and os.path.exists(original_path_in_cache) and original_path_in_cache.lower().endswith(".flac"):
            try:
                shutil.copy2(original_path_in_cache, output_filepath)
                print(f"Successfully copied original FLAC from cache: {original_path_in_cache} to {output_filepath}")
                return
            except Exception as e_copy:
                print(f"Warning: Failed to copy from cache path '{original_path_in_cache}': {e_copy}. Will try saving from array.")
        
        # 방법 2: 오디오 배열을 FLAC으로 저장 (캐시 경로가 없거나 FLAC이 아닐 경우)
        if audio_array is not None and sampling_rate is not None:
            try:
                sf.write(output_filepath, audio_array, samplerate=sampling_rate, format='FLAC', subtype='PCM_16')
                print(f"Successfully saved audio array to FLAC: {output_filepath}")
            except Exception as e_write:
                print(f"Error writing FLAC file for ID '{youtube_id}': {e_write}")
        else:
            print(f"Error: Cannot save FLAC for ID '{youtube_id}' as audio array or sampling rate is unavailable and direct copy failed.")

    except IndexError:
        print(f"Error: Item index {item_index} is out of bounds for the loaded dataset split.")
    except Exception as e:
        print(f"An unexpected error occurred while processing item at index {item_index}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a single audio file from a Hugging Face dataset.")
    parser.add_argument('--hf_item_index', type=int, required=True,
                        help="The 'hf_item_index' of the audio file to extract.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the extracted FLAC audio file.")
    parser.add_argument('--hf_dataset_name', type=str, default="agkphysics/AudioSet",
                        help="Name of the Hugging Face dataset (default: agkphysics/AudioSet).")
    parser.add_argument('--hf_config_name', type=str, default="balanced",
                        choices=["balanced", "unbalanced"],
                        help="Dataset configuration/name on Hugging Face (default: balanced).")
    parser.add_argument('--hf_split_name', type=str, default="train",
                        choices=["train", "test"], # agkphysics/AudioSet의 'balanced'와 'unbalanced' 설정은 'train', 'test' 스플릿을 가짐
                        help="Dataset split name on Hugging Face (default: train).")
    
    args = parser.parse_args()

    extract_single_audio_from_hf(
        hf_dataset_name=args.hf_dataset_name,
        hf_config_name=args.hf_config_name,
        hf_split_name=args.hf_split_name,
        item_index=args.hf_item_index,
        output_dir=args.output_dir
    )
