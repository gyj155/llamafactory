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
3D Feature Processing Modules

This module provides modular feature processing methods for integrating
3D spatial information (world coordinates, depth) into vision-language models.
"""

import torch
import torch.nn as nn


class BaseFeatureProcessor(nn.Module):
    """
    Base feature processor that performs no modifications.
    
    This serves as a passthrough for models that don't use 3D features.
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config
    
    def forward(self, image_features, world_coords=None, depths=None, **kwargs):
        """
        Process image features without modification.
        
        Args:
            image_features: Vision encoder outputs
            world_coords: 3D world coordinates (optional, ignored)
            depths: Depth maps (optional, ignored)
            **kwargs: Additional arguments (ignored)
        
        Returns:
            image_features: Unmodified features
        """
        return image_features


class PositionEmbeddingSine3D(nn.Module):
    """
    3D Sinusoidal Positional Encoding.
    
    Adapted from the position encoding in "Attention is All You Need",
    extended to 3D coordinates for spatial understanding.
    """
    
    def __init__(self, embedding_size, temperature=10000, n_points=1):
        """
        Initialize 3D positional encoding.
        
        Args:
            embedding_size: Size of the embedding dimension
            temperature: Temperature parameter for sinusoidal encoding
            n_points: Number of points per location (default 1)
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.temperature = temperature
        self.n_points = n_points
    
    def forward(self, x):
        """
        Compute sinusoidal positional encoding for 3D coordinates.
        
        Args:
            x: 3D coordinates (B, N, 3) or (B, N, n_points, 3)
        
        Returns:
            pos: Positional encoding (B, N, embedding_size)
        """
        num_feats = self.embedding_size // (3 * self.n_points)
        
        if self.n_points > 1 and x.dim() == 4:
            x = x.flatten(1, 2)
        
        B, N, _ = x.shape
        
        dim_t = torch.arange(num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_feats)
        
        pos_x = x[:, :, 0][..., None] / dim_t
        pos_y = x[:, :, 1][..., None] / dim_t
        pos_z = x[:, :, 2][..., None] / dim_t
        
        if num_feats % 2 != 0:
            pos_x = torch.cat([pos_x, torch.zeros(B, N, 1, device=pos_x.device)], dim=-1)
            pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)[..., :-1]
            pos_y = torch.cat([pos_y, torch.zeros(B, N, 1, device=pos_y.device)], dim=-1)
            pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)[..., :-1]
            pos_z = torch.cat([pos_z, torch.zeros(B, N, 1, device=pos_z.device)], dim=-1)
            pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)[..., :-1]
        else:
            pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
            pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
            pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)
        
        pos = torch.cat([pos_x, pos_y, pos_z], dim=2)
        
        if self.n_points > 1:
            pos = pos.view(B, N // self.n_points, self.n_points * 3 * num_feats)
        
        out = torch.zeros((B, N // self.n_points, self.embedding_size), dtype=x.dtype, device=x.device)
        out[:, :, :pos.shape[2]] = pos
        
        return out


class Sin3DPEProcessor(nn.Module):
    """
    Feature processor using sinusoidal 3D positional encoding.
    
    Adds 3D spatial awareness to image features by encoding world coordinates
    with sinusoidal functions and adding them to the feature representations.
    """
    
    def __init__(self, config):
        """
        Initialize sinusoidal 3D PE processor.
        
        Args:
            config: Model configuration with position_encoding_dim
        """
        super().__init__()
        self.config = config
        self.position_encoder = PositionEmbeddingSine3D(
            embedding_size=config.position_encoding_dim,
            temperature=10000,
            n_points=1
        )
    
    def forward(self, image_features, world_coords=None, depths=None, **kwargs):
        """
        Process image features with 3D positional encoding.
        
        Args:
            image_features: Vision encoder outputs (B, L, D) where L is sequence length
            world_coords: 3D world coordinates (B, V, H, W, 3)
            depths: Depth maps (B, V, H, W) - optional, not used
            **kwargs: Additional arguments
        
        Returns:
            image_features: Features enhanced with 3D positional encoding
        """
        if world_coords is None:
            return image_features
        
        # Reshape world_coords to match feature dimensions
        # world_coords: (B, V, H, W, 3) -> (B, V*H*W, 3)
        B, V, H, W, _ = world_coords.shape
        coords_flat = world_coords.view(B, V * H * W, 3)
        
        # Compute positional encoding
        pos_encoding = self.position_encoder(coords_flat)  # (B, V*H*W, D)
        
        # Match the sequence length of image_features
        # If image_features is (B, L, D) and L != V*H*W due to patch embedding,
        # we need to handle this mismatch
        feat_seq_len = image_features.shape[1]
        pos_seq_len = pos_encoding.shape[1]
        
        if feat_seq_len != pos_seq_len:
            # Interpolate or pool positional encoding to match feature length
            # For now, we'll use adaptive average pooling
            pos_encoding = pos_encoding.permute(0, 2, 1)  # (B, D, V*H*W)
            pos_encoding = nn.functional.adaptive_avg_pool1d(pos_encoding, feat_seq_len)
            pos_encoding = pos_encoding.permute(0, 2, 1)  # (B, L, D)
        
        # Ensure embedding dimension matches
        if pos_encoding.shape[-1] != image_features.shape[-1]:
            # Project positional encoding to match feature dimension
            if not hasattr(self, 'pos_proj'):
                self.pos_proj = nn.Linear(
                    pos_encoding.shape[-1],
                    image_features.shape[-1],
                    device=image_features.device,
                    dtype=image_features.dtype
                )
            pos_encoding = self.pos_proj(pos_encoding)
        
        # Add positional encoding to features
        enhanced_features = image_features + pos_encoding
        
        return enhanced_features


class MLP3DPEProcessor(nn.Module):
    """
    Feature processor using MLP-based 3D positional encoding.
    
    Uses a learnable MLP to encode 3D world coordinates into feature space.
    """
    
    def __init__(self, config):
        """
        Initialize MLP-based 3D PE processor.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        hidden_dim = config.position_encoding_dim
        
        # MLP for encoding 3D coordinates
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.text_config.hidden_size)
        )
    
    def forward(self, image_features, world_coords=None, depths=None, **kwargs):
        """
        Process image features with MLP-based 3D encoding.
        
        Args:
            image_features: Vision encoder outputs (B, L, D)
            world_coords: 3D world coordinates (B, V, H, W, 3)
            depths: Depth maps (B, V, H, W) - optional, not used
            **kwargs: Additional arguments
        
        Returns:
            image_features: Features enhanced with MLP-encoded 3D information
        """
        if world_coords is None:
            return image_features
        
        # Reshape world_coords to match feature dimensions
        B, V, H, W, _ = world_coords.shape
        coords_flat = world_coords.view(B, V * H * W, 3)
        
        # Encode coordinates with MLP
        coord_encoding = self.coord_encoder(coords_flat)  # (B, V*H*W, D)
        
        # Match sequence length
        feat_seq_len = image_features.shape[1]
        coord_seq_len = coord_encoding.shape[1]
        
        if feat_seq_len != coord_seq_len:
            coord_encoding = coord_encoding.permute(0, 2, 1)
            coord_encoding = nn.functional.adaptive_avg_pool1d(coord_encoding, feat_seq_len)
            coord_encoding = coord_encoding.permute(0, 2, 1)
        
        # Add encoded coordinates to features
        enhanced_features = image_features + coord_encoding
        
        return enhanced_features


def get_feature_processor(config):
    """
    Factory function to create feature processor based on configuration.
    
    Args:
        config: Model configuration with feature_processing_method
    
    Returns:
        Feature processor instance
    
    Raises:
        ValueError: If feature_processing_method is unknown
    """
    method = getattr(config, 'feature_processing_method', 'base')
    
    if method == 'base':
        return BaseFeatureProcessor(config)
    elif method == 'sin3d_pe':
        return Sin3DPEProcessor(config)
    elif method == 'mlp_pe':
        return MLP3DPEProcessor(config)
    else:
        raise ValueError(
            f"Unknown feature processing method: {method}. "
            f"Supported methods: 'base', 'sin3d_pe', 'mlp_pe'"
        )

