#!/usr/bin/env python3
"""
Data preprocessing script for 3D VLM training.

This script:
1. Reads datasets defined in multi.yaml
2. Samples frames from posed_images for each video
3. Converts to ShareGPT format with multi-image support
4. Extracts and saves scene metadata for training
"""

import os
import json
import pickle
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np


def load_scene_metadata(embodiedscan_dir):
    """
    Load scene metadata from embodiedscan pickle files.
    
    Returns:
        dict: scene_name -> metadata mapping
    """
    scene_metadata = {}
    
    for split in ["train", "val", "test"]:
        pkl_path = os.path.join(embodiedscan_dir, f"embodiedscan_infos_{split}.pkl")
        if not os.path.exists(pkl_path):
            print(f"Warning: {pkl_path} not found, skipping {split}")
            continue
        
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)["data_list"]
            for item in data:
                # item["sample_idx"]: "scannet/scene0415_00"
                if item["sample_idx"].startswith("scannet"):
                    scene_name = item["sample_idx"].split("/")[-1]  # e.g., scene0415_00
                    scene_metadata[scene_name] = {
                        'axis_align_matrix': item.get('axis_align_matrix'),
                        'depth_cam2img': item.get('depth_cam2img'),
                        'images': item.get('images', []),
                    }
    
    print(f"Loaded metadata for {len(scene_metadata)} scenes")
    return scene_metadata


def sample_frames_uniform(posed_images_dir, scene_name, num_frames=32):
    """
    Uniformly sample frames from a scene's posed_images directory.
    
    Args:
        posed_images_dir: Base directory for posed_images
        scene_name: Scene name (e.g., scene0011_00)
        num_frames: Number of frames to sample
        
    Returns:
        list: List of sampled frame paths (relative to posed_images_dir)
    """
    scene_dir = os.path.join(posed_images_dir, scene_name)
    
    if not os.path.exists(scene_dir):
        print(f"Warning: Scene directory {scene_dir} not found")
        return []
    
    # Get all jpg files
    all_frames = sorted([f for f in os.listdir(scene_dir) if f.endswith('.jpg')])
    
    if len(all_frames) == 0:
        print(f"Warning: No frames found in {scene_dir}")
        return []
    
    # Uniform sampling
    if len(all_frames) <= num_frames:
        sampled_frames = all_frames
    else:
        indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
        sampled_frames = [all_frames[i] for i in indices]
    
    # Return relative paths
    return [os.path.join(scene_name, frame) for frame in sampled_frames]


def process_single_item(item, posed_images_dir, scene_metadata, num_frames=32):
    """
    Process a single data item.
    
    Args:
        item: Original data item
        posed_images_dir: Base directory for posed_images
        scene_metadata: Scene metadata dict
        num_frames: Number of frames to sample
        
    Returns:
        dict: Processed item in sharegpt format
    """
    # Extract scene name from video field
    video = item.get('video', '')
    if not video.startswith('scannet/'):
        print(f"Warning: Unexpected video format: {video}")
        return None
    
    scene_name = video.split('/')[-1]  # e.g., scene0011_00
    
    # Sample frames
    sampled_frames = sample_frames_uniform(posed_images_dir, scene_name, num_frames)
    
    if len(sampled_frames) == 0:
        print(f"Warning: No frames sampled for {scene_name}, skipping item {item.get('id')}")
        return None
    
    # Assert we have exactly the expected number of frames

    
    # Convert conversations to messages (alpaca -> sharegpt)
    # Replace single <image> token with multiple <image> tokens (one per frame)
    messages = []
    image_tokens = ' '.join(['<image>'] * len(sampled_frames))
    
    for conv in item.get('conversations', []):
        role = 'user' if conv.get('from') == 'human' else 'assistant'
        content = conv.get('value', '')
        
        # Replace <image> token in the first user message
        if role == 'user' and '<image>' in content:
            # Assert that there is at least one <image> token
            assert '<image>' in content, f"No <image> token found in content: {content}"
            # Replace the first <image> with all image tokens
            content = content.replace('<image>', image_tokens, 1)
        
        messages.append({
            'content': content,
            'role': role
        })
    
    # Count <image> tokens in all messages
    total_image_tokens = sum(msg['content'].count('<image>') for msg in messages)
    assert total_image_tokens == len(sampled_frames), \
        f"Number of <image> tokens ({total_image_tokens}) must match number of images ({len(sampled_frames)})"
    
    # Create sharegpt format item
    processed_item = {
        'messages': messages,
        'images': [f"posed_images/{frame}" for frame in sampled_frames],  # Paths relative to data-3dllm
        'metadata': {
            **item.get('metadata', {}),
            'scene_name': scene_name,
            'original_video': video,
            'num_frames': len(sampled_frames),
        }
    }
    
    # Check if scene metadata exists
    if scene_name not in scene_metadata:
        print(f"Warning: No metadata found for scene {scene_name}")
    
    return processed_item


def process_json_file(json_path, posed_images_dir, scene_metadata, num_frames=32):
    """
    Process a single JSON dataset file.
    
    Args:
        json_path: Path to JSON file
        posed_images_dir: Base directory for posed_images
        scene_metadata: Scene metadata dict
        num_frames: Number of frames to sample
        
    Returns:
        list: List of processed items
    """
    print(f"\nProcessing {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items")
    
    processed_items = []
    skipped = 0
    
    for item in tqdm(data, desc=f"Processing {os.path.basename(json_path)}"):
        processed = process_single_item(item, posed_images_dir, scene_metadata, num_frames)
        if processed is not None:
            processed_items.append(processed)
        else:
            skipped += 1
    
    print(f"Processed {len(processed_items)} items, skipped {skipped} items")
    
    return processed_items


def process_yaml_config(yaml_path, posed_images_dir, scene_metadata, num_frames=32):
    """
    Process multiple datasets from a YAML config file.
    
    Args:
        yaml_path: Path to YAML config
        posed_images_dir: Base directory for posed_images
        scene_metadata: Scene metadata dict
        num_frames: Number of frames to sample
        
    Returns:
        list: Combined list of all processed items
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    datasets = config.get('datasets', [])
    all_processed_items = []
    
    print(f"Found {len(datasets)} datasets in {yaml_path}")
    
    for dataset_config in datasets:
        json_path = dataset_config.get('json_path')
        
        if not json_path:
            print(f"Warning: No json_path in dataset config: {dataset_config}")
            continue
        
        # Make path absolute if needed
        if not os.path.isabs(json_path):
            base_dir = os.path.dirname(yaml_path)
            json_path = os.path.join(base_dir, json_path)
        
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found: {json_path}")
            continue
        
        processed_items = process_json_file(json_path, posed_images_dir, scene_metadata, num_frames)
        all_processed_items.extend(processed_items)
    
    return all_processed_items


def main():
    parser = argparse.ArgumentParser(description='Prepare 3D VLM training dataset')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp3/yejie/llamafactory/data-3dllm',
        help='Base directory containing data'
    )
    parser.add_argument(
        '--yaml_config',
        type=str,
        default='multi.yaml',
        help='YAML config file (relative to data_dir)'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=32,
        help='Number of frames to sample per scene'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default='merged_training_data.json',
        help='Output JSON filename'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    yaml_path = data_dir / args.yaml_config
    posed_images_dir = data_dir / 'posed_images'
    embodiedscan_dir = data_dir / 'embodiedscan'
    output_path = data_dir / args.output_name
    metadata_output_path = data_dir / 'scene_metadata.pkl'
    
    # Validate paths
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")
    if not posed_images_dir.exists():
        raise FileNotFoundError(f"posed_images directory not found: {posed_images_dir}")
    if not embodiedscan_dir.exists():
        raise FileNotFoundError(f"embodiedscan directory not found: {embodiedscan_dir}")
    
    print("=" * 60)
    print("3D VLM Dataset Preparation")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"YAML config: {yaml_path}")
    print(f"Sampling {args.num_frames} frames per scene")
    print("=" * 60)
    
    # Load scene metadata
    print("\nStep 1: Loading scene metadata from embodiedscan...")
    scene_metadata = load_scene_metadata(str(embodiedscan_dir))
    
    # Process datasets
    print("\nStep 2: Processing datasets...")
    all_items = process_yaml_config(
        str(yaml_path),
        str(posed_images_dir),
        scene_metadata,
        args.num_frames
    )
    
    # Save processed dataset
    print(f"\nStep 3: Saving processed dataset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(all_items, f, indent=2)
    
    print(f"Saved {len(all_items)} items")
    
    # Save scene metadata
    print(f"\nStep 4: Saving scene metadata to {metadata_output_path}...")
    with open(metadata_output_path, 'wb') as f:
        pickle.dump(scene_metadata, f)
    
    print(f"Saved metadata for {len(scene_metadata)} scenes")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total items: {len(all_items)}")
    
    # Count unique scenes
    unique_scenes = set(item['metadata']['scene_name'] for item in all_items)
    print(f"Unique scenes: {len(unique_scenes)}")
    
    # Count images per item
    total_images = sum(len(item.get('images', [])) for item in all_items)
    print(f"Total images: {total_images}")
    print(f"Average images per item: {total_images / len(all_items) if all_items else 0:.1f}")
    
    # Count by dataset
    dataset_counts = {}
    for item in all_items:
        dataset = item['metadata'].get('dataset', 'unknown')
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    print("\nItems per dataset:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"  {dataset}: {count}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print(f"\nOutput files:")
    print(f"  Dataset: {output_path}")
    print(f"  Metadata: {metadata_output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
"""
{'messages': [{'content': '<image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> I stand looking out of the window in thought and a radiator is right in front of me. What color is the desk to my right? Answer the question using a single word or phrase.', 'role': 'user'}, {'content': 'brown', 'role': 'assistant'}], 'images': ['posed_images/scene0380_00/00000.jpg', 'posed_images/scene0380_00/00020.jpg', 'posed_images/scene0380_00/00050.jpg', 'posed_images/scene0380_00/00080.jpg', 'posed_images/scene0380_00/00110.jpg', 'posed_images/scene0380_00/00140.jpg', 'posed_images/scene0380_00/00170.jpg', 'posed_images/scene0380_00/00200.jpg', 'posed_images/scene0380_00/00230.jpg', 'posed_images/scene0380_00/00260.jpg', 'posed_images/scene0380_00/00290.jpg', 'posed_images/scene0380_00/00320.jpg', 'posed_images/scene0380_00/00350.jpg', 'posed_images/scene0380_00/00380.jpg', 'posed_images/scene0380_00/00410.jpg', 'posed_images/scene0380_00/00440.jpg', 'posed_images/scene0380_00/00470.jpg', 'posed_images/scene0380_00/00500.jpg', 'posed_images/scene0380_00/00530.jpg', 'posed_images/scene0380_00/00560.jpg', 'posed_images/scene0380_00/00590.jpg', 'posed_images/scene0380_00/00620.jpg', 'posed_images/scene0380_00/00650.jpg', 'posed_images/scene0380_00/00680.jpg', 'posed_images/scene0380_00/00710.jpg', 'posed_images/scene0380_00/00740.jpg', 'posed_images/scene0380_00/00770.jpg', 'posed_images/scene0380_00/00800.jpg', 'posed_images/scene0380_00/00830.jpg', 'posed_images/scene0380_00/00860.jpg', 'posed_images/scene0380_00/00890.jpg', 'posed_images/scene0380_00/00920.jpg'], 'metadata': {'dataset': 'sqa3d', 'question_type': 'what', 'scene_name': 'scene0380_00', 'original_video': 'scannet/scene0380_00', 'num_frames': 32}}
"""