import json
import argparse

def filter_sonyc_labels(input_filename, output_filename):
    """
    JSON 파일에서 '/m/sonyc_'로 시작하는 레이블을 가진 데이터를 제외하고
    새로운 JSON 파일로 저장합니다.

    Args:
        input_filename (str): 입력 JSON 파일 경로
        output_filename (str): 출력 JSON 파일 경로
    """
    try:
        # 입력 JSON 파일 읽기
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filtered_data = []
        # 'data' 키가 있는지 확인
        if 'data' in data and isinstance(data['data'], list):
            for item in data['data']:
                # 'labels' 키가 있고 문자열인지 확인
                if 'labels' in item and isinstance(item['labels'], str):
                    labels = item['labels'].split(',')
                    # '/m/sonyc_'로 시작하는 레이블이 없는 경우만 필터링된 리스트에 추가
                    if not any(label.strip().startswith('/m/sonyc_') for label in labels):
                        filtered_data.append(item)
                else:
                    # 'labels' 키가 없거나 문자열이 아닌 경우 원본 데이터를 그대로 추가 (선택 사항)
                    filtered_data.append(item)

            # 필터링된 데이터로 새 딕셔너리 생성
            new_data = {"data": filtered_data}

            # 새 JSON 파일로 저장 (indent=4 적용)
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)

            print(f"필터링된 데이터가 '{output_filename}'으로 성공적으로 저장되었습니다.")

        else:
            print(f"오류: '{input_filename}' 파일에 'data' 키가 없거나 리스트 형식이 아닙니다.")

    except FileNotFoundError:
        print(f"오류: '{input_filename}' 파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        print(f"오류: '{input_filename}' 파일이 유효한 JSON 형식이 아닙니다.")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")

def main():
    """스크립트의 메인 함수"""
    parser = argparse.ArgumentParser(description="JSON 파일에서 특정 레이블('/m/sonyc_')을 가진 데이터를 필터링합니다.")

    # 입력 파일 인자 추가
    parser.add_argument('-i', '--input',
                        type=str,
                        required=True,
                        help='입력 JSON 파일 경로')

    # 출력 파일 인자 추가
    parser.add_argument('-o', '--output',
                        type=str,
                        required=True,
                        help='필터링된 데이터를 저장할 출력 JSON 파일 경로')

    # 명령줄 인자 파싱
    args = parser.parse_args()

    # 필터링 함수 호출
    filter_sonyc_labels(args.input, args.output)

# 스크립트가 직접 실행될 때 main 함수 호출
if __name__ == "__main__":
    main()