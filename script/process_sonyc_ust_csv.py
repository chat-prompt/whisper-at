#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Process SONYC-UST CSV annotations file for Whisper-AT training
# Modified to use the entire SONYC-UST dataset for training

import os
import argparse
import csv
import pandas as pd
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Process SONYC-UST annotations CSV for Whisper-AT training')
    parser.add_argument('--annotations_csv', type=str, required=True, 
                        help='Path to SONYC-UST annotations.csv file')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing SONYC-UST audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed files')
    parser.add_argument('--class_mapping', type=str, required=True,
                        help='Path to class mapping JSON file (SONYC to AudioSet)')
    parser.add_argument('--audioset_classes_csv', type=str, required=True,
                        help='Path to AudioSet class_labels_indices.csv file')
    parser.add_argument('--verified_only', action='store_true',
                        help='Use only verified annotations (annotator_id=0)')
 
    return parser.parse_args()

def load_class_mapping(mapping_file):
    """
    Load the mapping between SONYC-UST classes and AudioSet classes
    """
    with open(mapping_file, 'r') as f:
        return json.load(f)

def load_audioset_classes(audioset_csv):
    """
    Load AudioSet class information
    """
    df = pd.read_csv(audioset_csv)
    class_map = {}
    index_to_mid = {}
    
    for _, row in df.iterrows():
        display_name = row['display_name'].strip('"')
        class_map[display_name] = int(row['index'])
        index_to_mid[int(row['index'])] = row['mid']
    
    return class_map, index_to_mid

def process_annotations(annotations_df, class_mapping, audioset_classes, verified_only=False, include_all_audioset=True):
    """
    Process annotations to create a mapping from audio files to their labels
    Modified to include all AudioSet classes
    """
    # Filter to only use verified annotations if requested
    if verified_only:
        annotations_df = annotations_df[annotations_df['annotator_id'] == 0]
    
    # Get all columns that indicate class presence
    presence_cols = [col for col in annotations_df.columns if col.endswith('_presence')]
    
    # Create a mapping from fine class name to its column
    fine_class_to_col = {}
    for col in presence_cols:
        if '-' in col:
            parts = col.split('_')[0].split('-')
            if len(parts) == 2 and parts[1] != 'X':  # Skip uncertain classes
                coarse_id, fine_id = parts
                fine_name = col.split('_')[1]
                fine_class_to_col[fine_name] = col
    
    # Create a mapping from audio filenames to labels
    audio_to_labels = defaultdict(set)
    audio_to_split = {}
    
    # Group by audio_filename to combine annotations
    grouped = annotations_df.groupby('audio_filename')
    
    for audio_filename, group in grouped:
        # Record the split for this file
        split = group['split'].iloc[0]
        audio_to_split[audio_filename] = split
        
        # Process each presence column
        for fine_name, col in fine_class_to_col.items():
            # Check if any annotator marked this class as present
            if (group[col] == 1).any():
                # If the fine class has a mapping to AudioSet, use that index
                if fine_name in class_mapping and class_mapping[fine_name] is not None:
                    audioset_class = class_mapping[fine_name]
                    if audioset_class in audioset_classes:
                        # Add the matching AudioSet class index
                        audio_to_labels[audio_filename].add(audioset_classes[audioset_class])
                else:
                    # This is a new class not in AudioSet
                    audio_to_labels[audio_filename].add(f"sonyc_{fine_name}")
    
    return audio_to_labels, audio_to_split

def create_extended_class_list(audio_to_labels, output_dir):
    """
    Create an extended class list including new SONYC-UST classes
    """
    # Find all unique label values
    all_labels = set()
    for labels in audio_to_labels.values():
        all_labels.update(labels)
    
    # Separate AudioSet indices and new SONYC classes
    audioset_indices = {label for label in all_labels if not isinstance(label, str) or not label.startswith('sonyc_')}
    new_sonyc_classes = {label for label in all_labels if isinstance(label, str) and label.startswith('sonyc_')}
    
    # Create mapping for new classes
    new_class_mapping = {}
    next_index = 527  # Start after AudioSet classes
    
    for class_name in sorted(new_sonyc_classes):
        new_class_mapping[class_name] = next_index
        next_index += 1
    
    # Save the mapping
    mapping_path = os.path.join(output_dir, "sonyc_new_class_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(new_class_mapping, f, indent=2)
    
    print(f"Created new class mapping with {len(new_class_mapping)} new classes")
    return new_class_mapping

def create_whisper_at_data(audio_to_labels, audio_to_split, audio_dir, new_class_mapping, index_to_mid, output_dir):
    """
    Create JSON files for Whisper-AT training
    Modified to use MID format labels consistent with AudioSet
    """
    train_data = []
    val_data = []
    test_data = []
    
    # Create reverse mapping from index to MID for new SONYC classes
    for class_name, index in new_class_mapping.items():
        display_name = class_name.replace('sonyc_', '')
        index_to_mid[index] = f"/m/sonyc_{display_name}"
    
    # Process each audio file
    for audio_filename, labels in audio_to_labels.items():
        # Check if the audio file exists
        audio_path = os.path.join(audio_dir, audio_filename)
        if not os.path.exists(audio_path):
            continue
        
        # Convert any string labels (new SONYC classes) to their numerical indices
        # and then get corresponding MIDs
        processed_labels = []
        for label in labels:
            if isinstance(label, str) and label.startswith('sonyc_'):
                label_index = new_class_mapping[label]
                processed_labels.append(index_to_mid[label_index])
            else:
                # label is already a numerical index for AudioSet
                processed_labels.append(index_to_mid[label])
        
        # Create the data entry
        entry = {
            "wav": audio_path,
            "labels": ",".join(processed_labels)
        }
        
        # Add to the appropriate split
        split = audio_to_split[audio_filename]
        if split == 'train':
            train_data.append(entry)
        elif split == 'validate':
            val_data.append(entry)
        elif split == 'test':
            test_data.append(entry)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the data files
    train_json = {"data": train_data}
    val_json = {"data": val_data}
    test_json = {"data": test_data}
    
    with open(os.path.join(output_dir, "sonyc_train.json"), 'w') as f:
        json.dump(train_json, f, indent=2)
    
    with open(os.path.join(output_dir, "sonyc_val.json"), 'w') as f:
        json.dump(val_json, f, indent=2)
    
    with open(os.path.join(output_dir, "sonyc_test.json"), 'w') as f:
        json.dump(test_json, f, indent=2)
    
    print(f"Created data files with {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test examples")
    return train_json, val_json, test_json

def create_updated_class_csv(audioset_csv, new_class_mapping, output_dir):
    """
    Create an extended class labels CSV including new SONYC classes
    """
    # Load original CSV
    df = pd.read_csv(audioset_csv)
    
    # Create rows for new SONYC classes
    new_rows = []
    for sonyc_class, index in new_class_mapping.items():
        display_name = sonyc_class.replace('sonyc_', '')
        new_rows.append({
            'index': index,
            'mid': f"/m/sonyc_{display_name}",
            'display_name': display_name  # Format consistent with AudioSet
        })
    
    # Add new rows
    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Save updated CSV
    output_path = os.path.join(output_dir, "class_labels_indices_extended.csv")
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)  # Avoid automatic quoting
    
    print(f"Extended class CSV file created with {len(new_rows)} new classes")
    return output_path

def generate_sample_weights(train_json, audioset_csv, new_class_mapping, output_dir):
    """
    Generate sample weights for the training data
    Modified to work with MID format labels
    """
    # Load the mapping from MID to index
    df = pd.read_csv(audioset_csv)
    mid_to_index = {}
    for _, row in df.iterrows():
        mid_to_index[row['mid']] = int(row['index'])
    
    # Add new SONYC classes to the mapping
    for class_name, index in new_class_mapping.items():
        display_name = class_name.replace('sonyc_', '')
        mid_to_index[f"/m/sonyc_{display_name}"] = index
    
    # Count occurrences of each class index
    label_count = np.zeros(max(mid_to_index.values()) + 1)
    
    for sample in train_json["data"]:
        sample_labels = sample['labels'].split(',')
        for label in sample_labels:
            if label in mid_to_index:
                label_idx = mid_to_index[label]
                label_count[label_idx] += 1
    
    # Calculate weights (1000 / count)
    label_weight = 1000.0 / (label_count + 0.01)
    
    # Calculate sample weights
    sample_weight = np.zeros(len(train_json["data"]))
    
    for i, sample in enumerate(train_json["data"]):
        sample_labels = sample['labels'].split(',')
        for label in sample_labels:
            if label in mid_to_index:
                label_idx = mid_to_index[label]
                # Sum weights for all classes in the sample
                sample_weight[i] += label_weight[label_idx]
    
    # Save weights
    weight_path = os.path.join(output_dir, "sonyc_train_weight.csv")
    np.savetxt(weight_path, sample_weight, delimiter=',')
    
    print(f"Generated sample weights saved to {weight_path}")
    return weight_path

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the annotations CSV file
    print(f"Loading annotations from {args.annotations_csv}")
    annotations_df = pd.read_csv(args.annotations_csv)
    
    # Load the class mapping and AudioSet classes
    class_mapping = load_class_mapping(args.class_mapping)
    audioset_classes, index_to_mid = load_audioset_classes(args.audioset_classes_csv)
    
    # Process the annotations
    print("Processing annotations...")
    audio_to_labels, audio_to_split = process_annotations(
        annotations_df, class_mapping, audioset_classes, 
        args.verified_only, args.include_all_audioset
    )
    
    # Create extended class list
    print("Creating extended class list...")
    new_class_mapping = create_extended_class_list(audio_to_labels, args.output_dir)
    
    # Create Whisper-AT data files
    print("Creating Whisper-AT data files...")
    train_json, val_json, test_json = create_whisper_at_data(
        audio_to_labels, audio_to_split, args.audio_dir, 
        new_class_mapping, index_to_mid, args.output_dir
    )
    
    # Create updated class CSV
    print("Creating updated class CSV...")
    updated_csv = create_updated_class_csv(args.audioset_classes_csv, new_class_mapping, args.output_dir)
    
    # Generate sample weights
    print("Generating sample weights...")
    weights_path = generate_sample_weights(train_json, args.audioset_classes_csv, new_class_mapping, args.output_dir)
    
    print("\nSUMMARY:")
    print(f"Total audio files processed: {len(audio_to_labels)}")
    print(f"New classes added: {len(new_class_mapping)}")
    print(f"Files created:")
    print(f"  - Training data: {os.path.join(args.output_dir, 'sonyc_train.json')}")
    print(f"  - Validation data: {os.path.join(args.output_dir, 'sonyc_val.json')}")
    print(f"  - Test data: {os.path.join(args.output_dir, 'sonyc_test.json')}")
    print(f"  - Class mapping: {os.path.join(args.output_dir, 'sonyc_new_class_mapping.json')}")
    print(f"  - Extended class CSV: {updated_csv}")
    print(f"  - Sample weights: {weights_path}")

if __name__ == "__main__":
    main()