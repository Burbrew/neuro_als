import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional


class WhiteNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class MeanDriftNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        _, C = x.shape
        noise = torch.randn(1, C) * self.std
        return x + noise

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")


class NeuralAugmentations(nn.Module):
    """
    Unified augmentation module for neural features.
    
    Consolidates all data augmentations applied during training:
    - White noise (per-sample, per-time, per-feature)
    - Baseline shift (per-sample, per-feature, constant across time)
    - Time masking (randomly mask contiguous time steps)
    - Feature masking (randomly mask contiguous feature channels)
    
    All augmentations are applied only during training (when model.training=True).
    
    Args:
        white_noise_std: Standard deviation of white noise added to all features (default: 0.8)
        baseline_shift_std: Standard deviation of baseline shift per feature channel (default: 0.2)
        time_mask_prob: Probability of applying time masking per sample (default: 0.0, disabled)
        time_mask_max_width: Maximum width of time mask in time steps (default: 0)
        feature_mask_prob: Probability of applying feature masking per sample (default: 0.0, disabled)
        feature_mask_max_width: Maximum width of feature mask in channels (default: 0)
        device: Device to run augmentations on (default: 'cuda')
    
    Example:
        >>> aug = NeuralAugmentations(
        ...     white_noise_std=0.8,
        ...     baseline_shift_std=0.2,
        ...     time_mask_prob=0.1,
        ...     time_mask_max_width=10,
        ...     feature_mask_prob=0.1,
        ...     feature_mask_max_width=5
        ... )
        >>> model.train()  # Enable training mode
        >>> augmented_X = aug(X)  # Apply augmentations
    """
    
    def __init__(
        self,
        white_noise_std: float = 0.8,
        baseline_shift_std: float = 0.2,
        time_mask_prob: float = 0.0,
        time_mask_max_width: int = 0,
        feature_mask_prob: float = 0.0,
        feature_mask_max_width: int = 0,
        device: str = "cuda",
    ):
        super().__init__()
        self.white_noise_std = white_noise_std
        self.baseline_shift_std = baseline_shift_std
        self.time_mask_prob = time_mask_prob
        self.time_mask_max_width = time_mask_max_width
        self.feature_mask_prob = feature_mask_prob
        self.feature_mask_max_width = feature_mask_max_width
        self.device = device
        
        # Validate parameters
        if time_mask_prob > 0 and time_mask_max_width <= 0:
            raise ValueError("time_mask_max_width must be > 0 when time_mask_prob > 0")
        if feature_mask_prob > 0 and feature_mask_max_width <= 0:
            raise ValueError("feature_mask_max_width must be > 0 when feature_mask_prob > 0")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to neural features.
        
        Args:
            x: Input tensor of shape [batch, time, features]
        
        Returns:
            Augmented tensor of same shape as input
        """
        # Only apply augmentations during training
        if not self.training:
            return x
        
        # White noise: add Gaussian noise to all features
        if self.white_noise_std > 0:
            x = x + torch.randn(x.shape, device=x.device) * self.white_noise_std
        
        # Baseline shift: add constant offset per feature channel (per sample)
        if self.baseline_shift_std > 0:
            # Shape: [batch, 1, features] - broadcast across time
            baseline_shift = torch.randn([x.shape[0], 1, x.shape[2]], device=x.device) * self.baseline_shift_std
            x = x + baseline_shift
        
        # Time masking: randomly mask contiguous time steps
        if self.time_mask_prob > 0 and self.time_mask_max_width > 0:
            x = self._apply_time_mask(x)
        
        # Feature masking: randomly mask contiguous feature channels
        if self.feature_mask_prob > 0 and self.feature_mask_max_width > 0:
            x = self._apply_feature_mask(x)
        
        return x
    
    def _apply_time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking: randomly mask contiguous time steps across all features.
        
        Args:
            x: Input tensor of shape [batch, time, features]
        
        Returns:
            Masked tensor (masked regions set to 0)
        """
        batch_size, time_steps, n_features = x.shape
        
        # Determine which samples to mask
        mask_samples = torch.rand(batch_size, device=x.device) < self.time_mask_prob
        
        for i in range(batch_size):
            if not mask_samples[i]:
                continue
            
            # Randomly choose mask width (1 to max_width)
            mask_width = torch.randint(1, self.time_mask_max_width + 1, (1,)).item()
            
            # Randomly choose start position
            max_start = time_steps - mask_width
            if max_start <= 0:
                continue  # Skip if sequence is too short
            
            start_pos = torch.randint(0, max_start + 1, (1,)).item()
            end_pos = start_pos + mask_width
            
            # Mask the time steps (set to 0 across all features)
            x[i, start_pos:end_pos, :] = 0
        
        return x
    
    def _apply_feature_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feature masking: randomly mask contiguous feature channels across all time.
        
        Args:
            x: Input tensor of shape [batch, time, features]
        
        Returns:
            Masked tensor (masked regions set to 0)
        """
        batch_size, time_steps, n_features = x.shape
        
        # Determine which samples to mask
        mask_samples = torch.rand(batch_size, device=x.device) < self.feature_mask_prob
        
        for i in range(batch_size):
            if not mask_samples[i]:
                continue
            
            # Randomly choose mask width (1 to max_width)
            mask_width = torch.randint(1, self.feature_mask_max_width + 1, (1,)).item()
            
            # Randomly choose start position
            max_start = n_features - mask_width
            if max_start <= 0:
                continue  # Skip if not enough features
            
            start_pos = torch.randint(0, max_start + 1, (1,)).item()
            end_pos = start_pos + mask_width
            
            # Mask the feature channels (set to 0 across all time steps)
            x[i, :, start_pos:end_pos] = 0
        
        return x
    
    def get_config_summary(self) -> str:
        """
        Get a summary string of the augmentation configuration.
        
        Returns:
            Human-readable summary of enabled augmentations
        """
        enabled = []
        if self.white_noise_std > 0:
            enabled.append(f"white_noise(std={self.white_noise_std})")
        if self.baseline_shift_std > 0:
            enabled.append(f"baseline_shift(std={self.baseline_shift_std})")
        if self.time_mask_prob > 0:
            enabled.append(f"time_mask(prob={self.time_mask_prob}, max_width={self.time_mask_max_width})")
        if self.feature_mask_prob > 0:
            enabled.append(f"feature_mask(prob={self.feature_mask_prob}, max_width={self.feature_mask_max_width})")
        
        if not enabled:
            return "No augmentations enabled"
        return ", ".join(enabled)
