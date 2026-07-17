import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from ultralytics.nn.modules.conv import Conv

__all__ = ["PAD"]

class Cut(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # 使用 YOLO 官方的 Conv 模块，它默认包含 Conv2d + BatchNorm2d + SiLU
        # 这里图上没有明确画出激活函数，但我们遵循 YOLO 的设计范式使用默认的 SiLU
        self.conv_fusion = Conv(c1 * 4, c2, k=1, s=1)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)  # 内部已包含 BN 和 Act
        return x

# Parallel Adaptive Downsample (PAD)
class PAD(nn.Module):
    """
    YOLOv11-RGBT PAD module.
    Args:
        c1 (int): number of input channels.
        c2 (int): number of output channels.
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.cut_c = Cut(c1, c2)

        # 1. 前置的 Group Conv: 使用官方 Conv，不使用激活函数 (act=False) 以保持与原图一致，但享受 BN 带来的稳定
        self.conv = Conv(c1, c2, k=3, s=1, g=c1, act=False)
        
        # 2. 中间的 DWConv + GELU + BatchNorm: 
        # YOLO Conv 默认是 SiLU，图上明确要求 GELU，所以我们传入 act=nn.GELU()
        self.conv_x = Conv(c2, c2, k=3, s=2, g=c2, act=nn.GELU())
        
        # 3. MaxPool 和 AvgPool
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 池化后的 BatchNorm
        self.batch_norm_m = nn.BatchNorm2d(c2)
        
        # 4. 尾部的 Fusion Conv: 使用官方 Conv 模块
        self.fusion = Conv(3 * c2, c2, k=1, s=1)

    def forward(self, x):  # input: x = [B, C, H, W]
        c = x  # c = [B, C, H, W]
        x = self.conv(x)  # x = [B, C, H, W] --> [B, out_channels, H, W]
        m = x  # m = [B, out_channels, H, W]

        # 1. CutD 分支 (底部 Adaptive Cut)
        c = self.cut_c(c)  # --> [B, out_channels, H/2, W/2]

        # 2. ConvD 分支 (中间 DWConv)
        # conv_x 内部已经包含了 DWConv2d + BatchNorm2d + GELU
        x = self.conv_x(x)  # --> [B, out_channels, H/2, W/2]

        # 3. MaxD + AvgD 分支 (顶部池化)
        m = self.max_pool(m) + self.avg_pool(m)  # --> [B, out_channels, H/2, W/2]
        m = self.batch_norm_m(m)

        # Concat + conv 融合
        out = torch.cat([c, x, m], dim=1)  # out = [B, 3 * out_channels, H/2, W/2]
        out = self.fusion(out)  # out = [B, out_channels, H/2, W/2]
        
        return out


if __name__ == "__main__":
    # 1. 实例化 PAD 模块
    in_ch = 64
    out_ch = 128
    pad_module = PAD(in_ch, out_ch)
    
    # 2. 构造一个测试用的输入 Tensor: [Batch_size, Channels, Height, Width]
    batch_size = 2
    height, width = 64, 64
    x = torch.randn(batch_size, in_ch, height, width)
    
    print(f"输入 Tensor 形状: {x.shape}")
    
    # 3. 前向传播
    output = pad_module(x)
    
    # 4. 打印输出形状并验证
    print(f"输出 Tensor 形状: {output.shape}")
    
    # 预期输出: 高和宽减半，通道数变为 out_channels
    expected_shape = (batch_size, out_ch, height // 2, width // 2)
    assert output.shape == expected_shape, f"形状错误！期望 {expected_shape} 但得到 {output.shape}"
    
    print("✅ 测试通过！模块输出形状符合预期。")
