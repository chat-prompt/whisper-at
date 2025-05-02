import csv
import json
import argparse
import numpy as np

def load_mid_to_index_mapping(csv_filepath):
    """
    CSV 파일에서 레이블 ID(mid)와 인덱스(index) 매핑 및 전체 클래스 수를 로드합니다.

    Args:
        csv_filepath (str): 레이블 매핑 CSV 파일 경로.

    Returns:
        tuple: (mid_to_index_map, num_classes)
               mid_to_index_map (dict): {mid: index} 형태의 딕셔너리.
               num_classes (int): 전체 레이블(클래스)의 개수.
               오류 발생 시 ({}, 0) 반환.
    """
    mid_to_index_map = {}
    num_classes = 0
    max_index = -1
    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader) # 헤더 읽고 건너뛰기
            if header != ['index', 'mid', 'display_name']:
                 print(f"경고: CSV 헤더가 예상과 다릅니다: {header}")

            for i, row in enumerate(reader):
                if len(row) >= 2:
                    try:
                        index = int(row[0].strip())
                        mid = row[1].strip()
                        mid_to_index_map[mid] = index
                        max_index = max(max_index, index)
                    except ValueError:
                        print(f"경고: 인덱스를 정수로 변환할 수 없습니다. 행 건너뜁니다: {row}")
                else:
                    print(f"경고: 형식이 잘못된 행을 건너뜁니다: {row}")
        # 클래스 개수는 가장 큰 인덱스 + 1
        num_classes = max_index + 1
        print(f"총 {len(mid_to_index_map)}개의 레이블 매핑 로드 완료. 전체 클래스 수: {num_classes}")
        return mid_to_index_map, num_classes
    except FileNotFoundError:
        print(f"오류: 레이블 매핑 파일 '{csv_filepath}'을(를) 찾을 수 없습니다.")
        return {}, 0
    except Exception as e:
        print(f"오류: 레이블 매핑 파일 로드 중 오류 발생: {e}")
        return {}, 0

def convert_to_one_hot(json_data, mid_to_index_map, num_classes):
    """
    JSON 데이터의 레이블을 멀티-레이블 원-핫 인코딩 벡터로 변환합니다.

    Args:
        json_data (dict): 변환할 원본 JSON 데이터 (파이썬 딕셔너리).
        mid_to_index_map (dict): load_mid_to_index_mapping으로 생성된 맵.
        num_classes (int): 전체 클래스 수 (원-핫 벡터의 길이).

    Returns:
        dict: 변환된 데이터. {"data": [{"wav": path, "one_hot_labels": [...]}, ...]} 형태.
              오류 발생 시 빈 데이터 딕셔너리 반환.
    """
    one_hot_encoded_data = []

    if not mid_to_index_map or num_classes <= 0:
        print("오류: 유효한 레이블 매핑 또는 클래스 수가 없어 변환 불가.")
        return {"data": []}

    if 'data' not in json_data or not isinstance(json_data['data'], list):
        print("오류: 입력 JSON 데이터 형식이 잘못되었습니다.")
        return {"data": []}

    for item in json_data['data']:
        if 'wav' in item and 'labels' in item and isinstance(item['labels'], str):
            wav_path = item['wav']
            label_ids = item['labels'].split(',')

            one_hot_vector = np.zeros(num_classes, dtype=int)

            for label_id in label_ids:
                clean_label_id = label_id.strip()
                if clean_label_id in mid_to_index_map:
                    index = mid_to_index_map[clean_label_id]
                    if 0 <= index < num_classes:
                        one_hot_vector[index] = 1
                    else:
                         print(f"경고: 유효하지 않은 인덱스({index}) 발견됨 (레이블: {clean_label_id}). 무시합니다.")
                else:
                    print(f"정보: 매핑되지 않는 레이블 ID '{clean_label_id}' 발견됨. 무시합니다.")

            one_hot_list = one_hot_vector.tolist()
            one_hot_encoded_data.append({
                "wav": wav_path,
                "one_hot_labels": one_hot_list
            })
        else:
            print(f"경고: 'wav' 또는 'labels' 키가 없거나 형식이 잘못된 항목 건너뜁니다: {item}")

    return {"data": one_hot_encoded_data}

def main():
    """스크립트 메인 함수"""
    parser = argparse.ArgumentParser(description="JSON 파일의 레이블을 CSV 매핑을 사용하여 멀티-레이블 원-핫 인코딩으로 변환합니다.")

    parser.add_argument('--csv', type=str, required=True,
                        help='레이블 매핑 CSV 파일 경로 (예: audioset_label.csv)')
    parser.add_argument('--input_json', type=str, required=True,
                        help='입력 JSON 파일 경로 (예: sonyc_test_filtered.json)')
    parser.add_argument('--output_json', type=str, required=True,
                        help='원-핫 인코딩된 결과를 저장할 출력 JSON 파일 경로')

    args = parser.parse_args()

    # 1. 레이블 매핑 로드
    mid_to_index, num_classes = load_mid_to_index_mapping(args.csv)

    if num_classes > 0: # 매핑 로드 성공 시
        try:
            # 2. 입력 JSON 로드
            with open(args.input_json, 'r', encoding='utf-8') as f:
                input_data = json.load(f)

            # 3. 원-핫 인코딩 변환
            one_hot_data = convert_to_one_hot(input_data, mid_to_index, num_classes)

            # 4. 결과 저장
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(one_hot_data, f, ensure_ascii=False, indent=4)
            print(f"원-핫 인코딩된 데이터가 '{args.output_json}'에 성공적으로 저장되었습니다.")

        except FileNotFoundError:
            print(f"오류: 입력 JSON 파일 '{args.input_json}'을(를) 찾을 수 없습니다.")
        except json.JSONDecodeError:
            print(f"오류: 입력 JSON 파일 '{args.input_json}'이(가) 유효한 형식이 아닙니다.")
        except Exception as e:
            print(f"오류: 처리 중 예외 발생: {e}")

if __name__ == "__main__":
    main()