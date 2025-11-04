# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
3D Vision-Language Model Data Collator

This module provides a custom data collator for 3D scene understanding tasks.
It extends the standard multimodal collator to load depth maps, camera poses,
and compute 3D world coordinates dynamically during training.
"""

import os
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from PIL import Image

from .collator import SFTDataCollatorWith4DAttentionMask


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ..hparams import DataArguments
    from ..model import Template


def unproject(intrinsics, poses, depths):
    """
    Unproject depth images to 3D world coordinates.
    
    Args:
        intrinsics: Camera intrinsic matrices (V, 4, 4)
        poses: Camera pose matrices (V, 4, 4)
        depths: Depth maps (V, H, W)
    
    Returns:
        world_coords: 3D world coordinates (V, H, W, 3)
    """
    V, H, W = depths.shape
    y = torch.arange(0, H, device=depths.device)
    x = torch.arange(0, W, device=depths.device)
    y, x = torch.meshgrid(y, x, indexing='ij')

    x = x.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)
    y = y.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)

    fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, H*W)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, H*W)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, H*W)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, H*W)

    z = depths.view(V, H*W) / 1000.0  # Convert depth to meters
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)

    world_coords = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)
    world_coords = world_coords[..., :3] / world_coords[..., 3].unsqueeze(-1)
    world_coords = world_coords.view(V, H, W, 3)

    return world_coords


@dataclass
class DataCollatorFor3DVision(SFTDataCollatorWith4DAttentionMask):
    """
    Data collator for 3D vision-language models.
    
    Extends the standard multimodal collator to support 3D scene understanding by:
    - Loading depth maps for each RGB image
    - Loading camera poses
    - Computing 3D world coordinates from depth and camera parameters
    - Adding 3D information to the batch for model consumption
    """
    
    data_args: Optional["DataArguments"] = None
    scene_metadata: Optional[dict] = None
    data_dir: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        
        # Load scene metadata if available
        if self.data_dir and self.scene_metadata is None:
            metadata_path = os.path.join(self.data_dir, 'scene_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.scene_metadata = pickle.load(f)
                print(f"Loaded scene metadata for {len(self.scene_metadata)} scenes")
    
    def load_depth_and_pose(self, image_path: str, data_dir: str):
        """
        Load depth map and camera pose for a given RGB image.
        
        Args:
            image_path: Path to RGB image (relative to data_dir, e.g., "posed_images/scene0011_00/00000.jpg")
            data_dir: Base data directory
        
        Returns:
            depth: Depth map as torch tensor (H, W)
            pose: Camera pose matrix as torch tensor (4, 4)
        """
        # Construct full paths
        full_image_path = os.path.join(data_dir, image_path)
        depth_path = full_image_path.replace('.jpg', '.png')
        pose_path = full_image_path.replace('.jpg', '.txt')
        
        # Load depth image
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth file not found: {depth_path}")
        
        with Image.open(depth_path) as depth_img:
            depth = np.array(depth_img).astype(np.int32)
            depth = torch.from_numpy(depth)
        
        # Load pose matrix
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Pose file not found: {pose_path}")
        
        pose = np.loadtxt(pose_path)
        pose = torch.from_numpy(pose)
        
        return depth, pose
    
    def compute_world_coordinates(
        self,
        image_paths: list[str],
        scene_name: str,
        data_dir: str
    ):
        """
        Compute 3D world coordinates for a batch of images from the same scene.
        
        Args:
            image_paths: List of RGB image paths
            scene_name: Scene identifier (e.g., "scene0011_00")
            data_dir: Base data directory
        
        Returns:
            world_coords: 3D world coordinates (V, H, W, 3)
            depths: Depth maps (V, H, W)
        """
        if self.scene_metadata is None or scene_name not in self.scene_metadata:
            # Return dummy data if metadata not available
            print(f"Warning: No metadata for scene {scene_name}, skipping 3D data")
            return None, None
        
        meta = self.scene_metadata[scene_name]
        axis_align_matrix = torch.from_numpy(np.array(meta['axis_align_matrix']))
        depth_intrinsic = torch.from_numpy(np.array(meta['depth_cam2img']))
        
        depths = []
        poses = []
        
        # Load depth and pose for each frame
        for image_path in image_paths:
            try:
                depth, pose = self.load_depth_and_pose(image_path, data_dir)
                depths.append(depth)
                poses.append(pose)
            except Exception as e:
                print(f"Warning: Failed to load depth/pose for {image_path}: {e}")
                return None, None
        
        depths = torch.stack(depths).float()  # (V, H, W)
        poses = torch.stack([axis_align_matrix @ pose for pose in poses]).float()  # (V, 4, 4)
        depth_intrinsic = depth_intrinsic.unsqueeze(0).repeat(len(image_paths), 1, 1).float()
        
        # Compute world coordinates
        world_coords = unproject(depth_intrinsic, poses, depths)
        
        return world_coords, depths
    
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        """
        Collate a batch of features with optional 3D data loading.
        
        Args:
            features: List of feature dicts from dataset
        
        Returns:
            Collated batch with optional 3D data (world_coords, depths, scene_names)
        """
        # Extract metadata and image paths before calling parent
        batch_metadata = []
        batch_image_paths = []
        
        for feature in features:
            # Try to get metadata from feature (passed through from dataset)
            if "_metadata" in feature:
                metadata = feature.pop("_metadata", None)
            else:
                metadata = feature.get("metadata", None)
            
            # Try to get image paths (these should be stored in the original dataset)
            if "_image_paths" in feature:
                image_paths = feature.pop("_image_paths", None)
            else:
                image_paths = None
            
            batch_metadata.append(metadata)
            batch_image_paths.append(image_paths)
        
        # Call parent collator
        batch = super().__call__(features)
        
        # Check if we should load 3D data
        use_3d = self.data_args and getattr(self.data_args, 'use_3d_features', False)
        
        if use_3d and self.data_dir:
            batch_world_coords = []
            batch_depths = []
            batch_scene_names = []
            
            for metadata, image_paths in zip(batch_metadata, batch_image_paths):
                if metadata and 'scene_name' in metadata and image_paths:
                    scene_name = metadata['scene_name']
                    
                    # Compute world coordinates
                    world_coords, depths = self.compute_world_coordinates(
                        image_paths, scene_name, self.data_dir
                    )
                    
                    if world_coords is not None:
                        batch_world_coords.append(world_coords)
                        batch_depths.append(depths)
                        batch_scene_names.append(scene_name)
                    else:
                        # Add dummy data if computation failed
                        batch_world_coords.append(None)
                        batch_depths.append(None)
                        batch_scene_names.append(None)
                else:
                    # No 3D data for this sample
                    batch_world_coords.append(None)
                    batch_depths.append(None)
                    batch_scene_names.append(None)
            
            # Only add to batch if we have valid 3D data
            if any(wc is not None for wc in batch_world_coords):
                # Filter out None values and stack
                valid_indices = [i for i, wc in enumerate(batch_world_coords) if wc is not None]
                
                if valid_indices:
                    # For now, we'll add the data as a list
                    # The model will need to handle potentially ragged tensors
                    batch["world_coords"] = batch_world_coords
                    batch["depths"] = batch_depths
                    batch["scene_names"] = batch_scene_names
                    batch["has_3d_data"] = [wc is not None for wc in batch_world_coords]
        
        return batch

