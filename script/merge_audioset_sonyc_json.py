# -*- coding: utf-8 -*-
import json
import os
import argparse
# pandas는 더 이상 필요 없으므로 제거
from tqdm import tqdm

def process_labels_to_mid_string(mid_input):
    """
    단일 MID 문자열, MID 문자열 리스트, 또는 쉼표로 구분된 MID 문자열을
    일관된 쉼표 구분 MID 문자열 형식으로 반환합니다.
    """
    mid_list = []

    if isinstance(mid_input, str):
        # 쉼표로 구분된 문자열일 수 있으므로 분리하고 공백 제거
        mid_list = [m.strip() for m in mid_input.split(',') if m.strip()]
    elif isinstance(mid_input, list):
        # 리스트 내 각 요소가 문자열인지 확인하고 공백 제거
        mid_list = [str(m).strip() for m in mid_input if isinstance(m, str) and m.strip()]
    else:
        tqdm.write(f"Warning: Unexpected label format '{mid_input}'. Expected string or list of MIDs. Skipping.")
        return "" # 빈 문자열 반환

    # 유효한 MID 형식인지 간단히 확인 (예: '/m/' 또는 '/t/'로 시작) - 선택 사항
    valid_mids = [mid for mid in mid_list if mid.startswith(('/m/', '/t/', '/g/'))]
    
    if len(valid_mids) != len(mid_list):
         tqdm.write(f"Warning: Some invalid MIDs found or filtered in '{mid_input}'. Result: {','.join(valid_mids)}")

    return ",".join(valid_mids)

def merge_datasets(sonyc_json_path, sonyc_feature_base_dir,
                   audioset_json_path,
                   output_json_path):
    """
    SONYC-UST JSON과 AudioSet JSON을 병합합니다.
    두 데이터셋의 MID 레이블 형식을 쉼표로 구분된 문자열로 통일합니다.
    특징 파일 경로를 절대 경로로 변환하여 새로운 JSON 파일을 생성합니다.
    """
    combined_data_list = []

    # 레이블 맵 로딩은 더 이상 필요 없음

    # 1. SONYC-UST 데이터 처리
    print(f"Processing SONYC-UST data from: {sonyc_json_path}")
    try:
        with open(sonyc_json_path, 'r') as f:
            sonyc_data = json.load(f)
        
        for entry in tqdm(sonyc_data.get("data", []), desc="SONYC-UST data"):
            original_wav_path = entry.get("wav")
            # SONYC JSON의 labels는 MID 문자열 또는 쉼표로 구분된 MID 문자열
            mid_labels_input = entry.get("labels") 

            if not original_wav_path or mid_labels_input is None:
                tqdm.write(f"Warning: Skipping SONYC entry due to missing 'wav' or 'labels': {entry}")
                continue

            if os.path.isabs(original_wav_path):
                abs_feature_path = original_wav_path
            elif sonyc_feature_base_dir:
                abs_feature_path = os.path.abspath(os.path.join(sonyc_feature_base_dir, original_wav_path))
            else:
                tqdm.write(f"Warning: SONYC wav path '{original_wav_path}' is not absolute and no base_dir provided. Using as is.")
                abs_feature_path = original_wav_path
            
            # SONYC MID 레이블을 쉼표 구분 문자열 형식으로 통일
            labels_mid_str = process_labels_to_mid_string(mid_labels_input)
            
            if not labels_mid_str:
                tqdm.write(f"Warning: No valid MIDs found for SONYC entry with labels '{mid_labels_input}'. Skipping entry.")
                continue

            combined_data_list.append({
                "wav": abs_feature_path,
                "labels": labels_mid_str # MID 문자열 사용
            })
    except FileNotFoundError:
        print(f"Error: SONYC JSON file not found at {sonyc_json_path}")
    except Exception as e:
        print(f"Error processing SONYC JSON file: {e}")

    # 2. AudioSet 데이터 처리
    print(f"\nProcessing AudioSet data from: {audioset_json_path}")
    try:
        with open(audioset_json_path, 'r') as f:
            audioset_data = json.load(f)
        
        audioset_json_dir = os.path.dirname(os.path.abspath(audioset_json_path))
            
        for entry in tqdm(audioset_data.get("data", []), desc="AudioSet data"):
            relative_feature_path = entry.get("feature_path") 
            mid_labels_input = entry.get("labels") # AudioSet은 MID 문자열 리스트였음

            if not relative_feature_path or mid_labels_input is None:
                tqdm.write(f"Warning: Skipping AudioSet entry due to missing 'feature_path' or 'labels': {entry}")
                continue
            
            abs_feature_path = os.path.abspath(os.path.join(audioset_json_dir, relative_feature_path))
            
            # AudioSet MID 리스트/문자열을 쉼표 구분 문자열 형식으로 통일
            labels_mid_str = process_labels_to_mid_string(mid_labels_input)
            
            if not labels_mid_str:
                tqdm.write(f"Warning: No valid MIDs found for AudioSet entry with MIDs {mid_labels_input}. Skipping entry.")
                continue

            combined_data_list.append({
                "wav": abs_feature_path,
                "labels": labels_mid_str # MID 문자열 사용
            })
    except FileNotFoundError:
        print(f"Error: AudioSet JSON file not found at {audioset_json_path}")
    except Exception as e:
        print(f"Error processing AudioSet JSON file: {e}")

    # 3. 병합된 데이터 저장
    output_full_data = {"data": combined_data_list}
    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(output_full_data, f, indent=2)
        print(f"\nSuccessfully merged {len(combined_data_list)} entries.")
        print(f"Combined data saved to: {output_json_path}")
    except Exception as e:
        print(f"Error saving combined JSON file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge SONYC-UST and AudioSet feature JSONs with absolute paths, keeping MID labels.")
    parser.add_argument('--sonyc_json', type=str, required=True,
                        help="Path to the SONYC-UST training data JSON file (e.g., sonyc_new_train.json).")
    parser.add_argument('--sonyc_feature_base_dir', type=str, default="",
                        help="(Optional) Base directory for SONYC feature files if paths in sonyc_json are relative. "
                             "If sonyc_json 'wav' paths are absolute, this can be omitted.")
    parser.add_argument('--audioset_json', type=str, required=True,
                        help="Path to the AudioSet feature metadata JSON file "
                             "(e.g., audioset_balanced-train_large-v1_all_features.json).")
    # --label_csv 인자 제거
    parser.add_argument('--output_json', type=str, required=True,
                        help="Path to save the combined training data JSON file (e.g., combined_train.json).")
    
    args = parser.parse_args()

    merge_datasets(args.sonyc_json,
                   args.sonyc_feature_base_dir,
                   args.audioset_json,
                   # label_csv 인자 제거됨
                   args.output_json)
