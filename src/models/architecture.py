"""
FOMO-like YOLOv8 Micro Model

A lightweight PyTorch implementation combining YOLOv8 backbone (C2f blocks)
with FOMO-style heatmap head for edge device object detection.

Input: 160x160 or 96x96
Output: heatmap of class probabilities (e.g., 20x20 for 160x160 input)
"""

import torch
import torch.nn as nn


class Conv(nn.Module):
    """Standard convolution block with Conv2d + BatchNorm + SiLU activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block used in C2f."""
    
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    C2f block from YOLOv8.
    Features cross-stage partial connections with bottleneck blocks.
    """
    
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, groups=1, expansion=0.5, stride=1):
        super().__init__()
        self.c = int(out_channels * expansion)
        
        # Main conv with optional stride for downsampling
        if stride == 2:
            self.cv1 = Conv(in_channels, 2 * self.c, 3, stride=2)
        else:
            self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        
        self.cv2 = Conv((2 + n) * self.c, out_channels, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, groups, expansion=1.0) for _ in range(n))
    
    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MicroYOLO(nn.Module):
    """
    FOMO-like YOLOv8 micro model for edge devices.
    
    Args:
        nc: Number of classes
        input_size: Input image size (160 or 96)
    """
    
    def __init__(self, nc=2, input_size=160):
        super().__init__()
        self.nc = nc
        self.input_size = input_size
        
        # Backbone (based on the provided config)
        self.stem = Conv(3, 16, 3, 2)           # [3, 16, 3, 2] -> 160→80 or 96→48
        self.stage1 = C2f(16, 32, n=3)          # [16, 32, 3] (no stride change)
        self.stage2 = C2f(32, 64, n=3, stride=2)  # [32, 64, 3, 2] -> 80→40 or 48→24
        self.stage3 = C2f(64, 128, n=3, stride=2) # [64, 128, 3, 2] -> 40→20 or 24→12
        self.stage4 = C2f(128, 160, n=3)        # [128, 160, 3] (no stride change)
        
        # FOMO Head (heatmap classifier)
        self.head_conv1 = Conv(160, 32, 1, 1)   # [160, 32, 1, 1]
        self.head_conv2 = nn.Conv2d(32, nc, 1, 1)  # [32, nc, 1, 1] - no activation yet
        self.sigmoid = nn.Sigmoid()             # Final activation
    
    def forward(self, x, return_logits=False):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
            return_logits: If True, return logits without sigmoid (for training)
        
        Returns:
            Heatmap predictions (with sigmoid if return_logits=False)
        """
        # Backbone
        x = self.stem(x)        # 160→80 or 96→48
        x = self.stage1(x)      # same size
        x = self.stage2(x)      # 80→40 or 48→24
        x = self.stage3(x)      # 40→20 or 24→12
        x = self.stage4(x)      # same size
        
        # Head
        x = self.head_conv1(x)
        x = self.head_conv2(x)  # Logits
        
        if return_logits:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return self.sigmoid(x)  # Return probabilities for inference

    
    def get_output_size(self):
        """Calculate output heatmap size based on input size."""
        return self.input_size // 8


if __name__ == "__main__":
    # Quick test
    model = MicroYOLO(nc=2, input_size=160)
    x = torch.randn(1, 3, 160, 160)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected: (1, 2, 20, 20)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
