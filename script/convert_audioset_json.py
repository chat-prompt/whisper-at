# -*- coding: utf-8 -*-
import json
import os
import argparse
from tqdm import tqdm

def process_labels_to_mid_string(mid_input):
    """
    단일 MID 문자열, MID 문자열 리스트, 또는 쉼표로 구분된 MID 문자열을
    일관된 쉼표 구분 MID 문자열 형식으로 반환합니다.
    (merge_json_datasets_py 스크립트의 함수와 동일)
    """
    mid_list = []

    if isinstance(mid_input, str):
        mid_list = [m.strip() for m in mid_input.split(',') if m.strip()]
    elif isinstance(mid_input, list):
        mid_list = [str(m).strip() for m in mid_input if isinstance(m, str) and m.strip()]
    else:
        tqdm.write(f"Warning: Unexpected label format '{mid_input}'. Expected string or list of MIDs. Skipping.")
        return ""

    valid_mids = [mid for mid in mid_list if mid.startswith(('/m/', '/t/', '/g/'))]
    
    if len(valid_mids) != len(mid_list):
         tqdm.write(f"Warning: Some invalid MIDs found or filtered in '{mid_input}'. Result: {','.join(valid_mids)}")

    return ",".join(valid_mids)

def convert_audioset_json_format(input_json_path, output_json_path):
    """
    extract_hf_audioset.py로 생성된 AudioSet JSON 파일의 형식을 변환합니다.
    - feature_path를 절대 경로로 변환하여 'wav' 필드에 저장합니다.
    - labels 리스트를 쉼표로 구분된 MID 문자열로 변환하여 'labels' 필드에 저장합니다.

    Args:
        input_json_path (str): 변환할 원본 AudioSet 특징 메타데이터 JSON 파일 경로.
        output_json_path (str): 변환된 결과 JSON 파일 저장 경로.
    """
    converted_data_list = []

    print(f"Processing AudioSet data from: {input_json_path}")
    try:
        with open(input_json_path, 'r') as f:
            audioset_data = json.load(f)
        
        # 입력 JSON 파일의 디렉토리 경로 (상대 경로의 기준점)
        input_json_dir = os.path.dirname(os.path.abspath(input_json_path))
            
        for entry in tqdm(audioset_data.get("data", []), desc="Converting AudioSet data"):
            relative_feature_path = entry.get("feature_path") 
            mid_labels_input = entry.get("labels") # MID 문자열 리스트 또는 문자열

            if not relative_feature_path or mid_labels_input is None:
                tqdm.write(f"Warning: Skipping entry due to missing 'feature_path' or 'labels': {entry}")
                continue
            
            # 특징 파일의 절대 경로 생성
            abs_feature_path = os.path.abspath(os.path.join(input_json_dir, relative_feature_path))
            
            # MID 레이블을 쉼표 구분 문자열 형식으로 통일
            labels_mid_str = process_labels_to_mid_string(mid_labels_input)
            
            if not labels_mid_str:
                tqdm.write(f"Warning: No valid MIDs found for entry with labels {mid_labels_input}. Skipping entry.")
                continue

            converted_data_list.append({
                "wav": abs_feature_path, # 출력 키는 "wav"
                "labels": labels_mid_str # 쉼표로 구분된 MID 문자열
            })
    except FileNotFoundError:
        print(f"Error: Input AudioSet JSON file not found at {input_json_path}")
        return # 파일이 없으면 종료
    except Exception as e:
        print(f"Error processing input AudioSet JSON file: {e}")
        return # 오류 발생 시 종료

    # 변환된 데이터 저장
    output_full_data = {"data": converted_data_list}
    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True) # 출력 디렉토리 생성
        with open(output_json_path, 'w') as f:
            json.dump(output_full_data, f, indent=2)
        print(f"\nSuccessfully converted {len(converted_data_list)} entries.")
        print(f"Converted data saved to: {output_json_path}")
    except Exception as e:
        print(f"Error saving converted JSON file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert AudioSet feature JSON format for training/validation.")
    parser.add_argument('--input_json', type=str, required=True,
                        help="Path to the input AudioSet feature metadata JSON file "
                             "(e.g., audioset_eval_large-v1_all_features.json from extract_hf_audioset.py).")
    parser.add_argument('--output_json', type=str, required=True,
                        help="Path to save the converted JSON file (e.g., audioset_eval_formatted.json).")
    
    args = parser.parse_args()

    convert_audioset_json_format(args.input_json,
                                 args.output_json)
