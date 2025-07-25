import os
import argparse

def find_small_files(directory, threshold_bytes):
    small_files = []
    print(f"Scanning directory: {directory}")
    print(f"Finding .npz files smaller than {threshold_bytes} bytes...")
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".npz"):
                filepath = os.path.join(directory, filename)
                try:
                    filesize = os.path.getsize(filepath)
                    if filesize < threshold_bytes:
                        small_files.append((filepath, filesize))
                except OSError as e:
                    print(f"Could not get size for {filepath}: {e}")
    except FileNotFoundError:
        print(f"Error: Directory not found: {directory}")
        return []
    return small_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find small .npz files in a directory.")
    parser.add_argument('--feature_dir', type=str, required=True, help="Directory containing .npz feature files.")
    # 정상 파일 크기가 약 3.5MB (3,500,000 바이트)이므로, 임계값을 1MB (1,000,000 바이트) 정도로 설정
    parser.add_argument('--size_threshold', type=int, default=1000000, help="File size threshold in bytes. Files smaller than this will be listed.")
    
    args = parser.parse_args()

    found_files = find_small_files(args.feature_dir, args.size_threshold)

    if found_files:
        print(f"\nFound {len(found_files)} .npz files smaller than {args.size_threshold} bytes:")
        for filepath, filesize in found_files:
            print(f"- {filepath} ({filesize} bytes)")
        print("\nConsider removing these files and their corresponding entries from your JSON data files.")
    else:
        print("\nNo .npz files smaller than the threshold found.")
        