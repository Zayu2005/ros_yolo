import torch
import torch.nn as nn

from .WTConv import WTConv

class SGWTConv(nn.Module):
    """
    Selective Gated Wavelet Transform Convolution (SGWTConv).
    Lightweight gating + Residual differential enhancement.
    
    Formula:
    enh = WTConv(x)
    gate = sigmoid(conv1x1(dwconv3x3(x)))
    out = x + alpha * gate * (enh - x)
    """
    def __init__(self, c1, c2, k=5, s=1, p=None, g=1, d=1, act=True, wt_levels=1, wt_type='db1'):
        """
        Initialize SGWTConv with given parameters.
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            wt_levels (int): Wavelet transform levels.
            wt_type (str): Wavelet type.
        """
        super().__init__()
        
        # Enhancement branch (WTConv)
        self.wtconv = WTConv(c1, c2, k=k, s=s, p=p, g=g, d=d, act=act, wt_levels=wt_levels, wt_type=wt_type)
        
        # Gate branch: dwconv3x3 -> conv1x1 -> sigmoid
        # Adding autopad logic from ultralytics if needed, but since we use padding=1 for k=3 it's fine.
        # But to be perfectly safe with any kernel size passed to WTConv, the gate shouldn't restrict to k=3 if not necessary.
        # However, as a lightweight gate, dwconv3x3 is standard.
        self.dwconv3x3 = nn.Conv2d(c1, c1, kernel_size=3, stride=s, padding=1, groups=c1, bias=False)
        self.conv1x1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Shortcut for residual connection if channels or spatial dims change
        if c1 != c2 or s != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(c2)
            )
        else:
            self.shortcut = nn.Identity()
            
        # Learnable parameter alpha (channel-wise scaling), initialized to 1.0
        # To avoid AMP issues and ensure it works correctly across devices, we register it as a Parameter correctly.
        self.alpha = nn.Parameter(torch.ones(1, c2, 1, 1))

    def forward(self, x):
        """Apply Selective Gated WTConv."""
        # 1. Enhancement features
        enh = self.wtconv(x)
        
        # 2. Lightweight gating
        gate = self.sigmoid(self.conv1x1(self.dwconv3x3(x)))
        
        # 3. Shortcut projection (if needed)
        x_proj = self.shortcut(x)
        
        # 4. Residual differential enhancement
        # Ensure alpha dtype matches x_proj dtype to avoid AMP issues
        alpha = self.alpha.to(x_proj.dtype)
        out = x_proj + alpha * gate * (enh - x_proj)
        
        return out

    def forward_fuse(self, x):
        """Apply fused forward pass if applicable (YOLO export/inference)."""
        # Note: If the model uses rep-style or BN fusion, forward_fuse might be called.
        enh = self.wtconv.forward_fuse(x) if hasattr(self.wtconv, 'forward_fuse') else self.wtconv(x)
        
        gate = self.sigmoid(self.conv1x1(self.dwconv3x3(x)))
        x_proj = self.shortcut(x)
        
        alpha = self.alpha.to(x_proj.dtype)
        out = x_proj + alpha * gate * (enh - x_proj)
        return out


if __name__ == '__main__':
    # Add root path to sys.path so we can run this script directly
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    
    input_tensor = torch.randn(3, 32, 64, 64)  # b c h w
    sgwtconv = SGWTConv(c1=32, c2=64, k=3, s=2)
    output = sgwtconv(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
