#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Extract Whisper features for SONYC-UST dataset using whisper package

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import whisper
import traceback

def parse_args():
    parser = argparse.ArgumentParser(description='Extract Whisper features for SONYC-UST dataset')
    parser.add_argument('--sonyc_data_json', type=str, required=True, 
                        help='Path to processed SONYC-UST data JSON file (sonyc_train.json, etc.)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save extracted features')
    parser.add_argument('--model_size', type=str, default='large-v2',
                        choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 
                                'medium', 'medium.en', 'large-v1', 'large-v2', 'large'],
                        help='Whisper model size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed error messages')
    return parser.parse_args()

def extract_features(data, model, output_dir, device, verbose=False):
    """
    Extract features for all audio files in the dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Keep track of processed files
    processed_data = []
    
    for idx, entry in enumerate(tqdm(data['data'], desc="Extracting features")):
        audio_path = entry['wav']
        labels = entry['labels']
        
        # Extract filename for saving
        filename = os.path.basename(audio_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npz")
        
        # Skip if already processed
        if os.path.exists(output_path):
            processed_data.append({
                "wav": os.path.splitext(filename)[0],
                "labels": labels
            })
            continue
        
        try:
            # Load and preprocess audio
            # Important: We need to load audio and pad it to 30 seconds as Whisper expects
            audio = whisper.load_audio(audio_path)
            
            # Whisper expects 30 seconds of audio
            # SONYC-UST files are 10 seconds, so we need to pad
            target_length = 30 * whisper.audio.SAMPLE_RATE
            padded_audio = np.zeros(target_length, dtype=np.float32)
            padded_audio[:len(audio)] = audio
            
            # Log mel spectrogram with padding
            mel = whisper.log_mel_spectrogram(padded_audio).to(device)
            
            # Process with model - use built-in encoder and extract features we need
            with torch.no_grad():
                # Run the encoder
                encoder_output = model.encoder(mel.unsqueeze(0))
                
                # Collect features from all layers - we need to hook into the model
                layer_features = []
                
                # We need to reprocess the audio to get all layer outputs
                x = mel.unsqueeze(0)
                
                # Apply initial convolutions
                x = model.encoder.conv1(x)
                x = torch.nn.functional.gelu(x)
                x = model.encoder.conv2(x)
                x = torch.nn.functional.gelu(x)
                x = x.permute(0, 2, 1)
                
                # Apply positional embedding
                x = x + model.encoder.positional_embedding.to(x.dtype)
                
                # Process through transformer blocks and collect all layer outputs
                for block_idx, block in enumerate(model.encoder.blocks):
                    x = block(x)
                    # Store the layer output
                    layer_output = x.clone()  # [batch, time, dim]
                    
                    # We only want the first 10 seconds (1/3 of the 30s window)
                    # After Whisper's 2x downsampling, 10s = 500 frames
                    first_10s = layer_output[:, :500, :]
                    
                    # Apply 20x time pooling to match Whisper-AT approach
                    # Reshape for pooling [batch, dim, time]
                    pooled = torch.nn.functional.avg_pool1d(
                        first_10s.permute(0, 2, 1),
                        kernel_size=20,
                        stride=20
                    ).permute(0, 2, 1)  # [batch, time/20, dim]
                    
                    layer_features.append(pooled)
            
            # Stack all layer outputs
            stacked_features = torch.stack(layer_features, dim=0)  # [n_layers, batch, time/20, dim]
            
            # Convert to numpy
            features_np = stacked_features.cpu().numpy()
            
            # Remove batch dimension: [n_layers, batch=1, time/20, dim] -> [n_layers, time/20, dim]
            features_np = features_np[:, 0, :, :]
            
            # Save as compressed npz
            np.savez_compressed(output_path, features_np)
            
            # Add to processed data list
            processed_data.append({
                "wav": os.path.splitext(filename)[0],
                "labels": labels
            })
            
            # Periodically clean up GPU memory
            if idx % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            if verbose:
                print(f"Error processing {audio_path}: {e}")
                traceback.print_exc()
            else:
                print(f"Error processing {audio_path}: {e}")
    
    # Save processed file list
    output_json = os.path.join(output_dir, "processed_data.json")
    with open(output_json, 'w') as f:
        json.dump({"data": processed_data}, f, indent=2)
    
    print(f"Saved {len(processed_data)} feature files to {output_dir}")
    print(f"Saved processed data list to {output_json}")
    
    return output_json

def main():
    args = parse_args()
    
    # Load the SONYC-UST data
    print(f"Loading data from {args.sonyc_data_json}")
    with open(args.sonyc_data_json, 'r') as f:
        data = json.load(f)
    
    # Load Whisper model using the package
    print(f"Loading Whisper model: {args.model_size}")
    model = whisper.load_model(args.model_size).to(args.device)
    model.eval()
    
    # Print model dimensions
    n_mels = model.dims.n_mels
    n_audio_state = model.dims.n_audio_state
    n_audio_layer = model.dims.n_audio_layer
    n_audio_ctx = model.dims.n_audio_ctx
    print(f"Model dimensions: {n_mels} mels, {n_audio_state} hidden dim, {n_audio_layer} layers, {n_audio_ctx} context length")
    
    # Extract features
    print(f"Extracting features for {len(data['data'])} audio files")
    output_json = extract_features(data, model, args.output_dir, args.device, args.verbose)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    print(f"Feature extraction completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()