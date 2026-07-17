import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentialFeatureEnhancement(nn.Module):
    """
    差分特征增强模块 (Differential Feature Enhancement, DFE)
    
    核心思想 (来自 DEGF-YOLO 论文):
    - 共同特征: RGB和IR共享的轮廓/形状信息 → 作为融合基础
    - 差分特征: 模态特有的互补信息 → 用于增强判别性
    - 问题: 直接融合会导致"跨模态污染"（退化模态的噪声干扰有效模态）
    - 解决: 先用差分信息增强目标区域，再抑制背景噪声
    
    适用场景:
    - DroneVehicle 航拍数据集 (背景复杂、小目标多)
    - 日夜交替场景 (模态质量动态变化)
    - OBB 旋转框检测任务
    
    Args:
        c1: 输入通道数 (单模态通道数)
        c2: 输出通道数 (可选，默认与c1相同)
        reduction: 通道降维比例
    """
    def __init__(self, c1, c2=None, reduction=16):
        super().__init__()
        
        channels = c1[0] if isinstance(c1, list) else c1
        out_channels = c2 if c2 is not None else channels
        
        reduction_c = max(1, channels // reduction)
        
        self.channels = channels
        self.out_channels = out_channels
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc_common = nn.Sequential(
            nn.Linear(channels * 2, reduction_c),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_c, channels),
            nn.Sigmoid()
        )
        
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False)
        )
        
        self.noise_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduction_c),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_c, channels),
            nn.Sigmoid()
        )
        
        self.proj = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()
        self.bn_out = nn.BatchNorm2d(out_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: list 包含两个特征图 [rgb_feat, ir_feat]
               每个特征图 shape: [B, C, H, W]
        
        Returns:
            output: 融合后的特征图 [B, C_out, H, W]
        """
        rgb_feat = x[0]
        ir_feat = x[1]
        
        f_common = (rgb_feat + ir_feat) / 2.0
        
        f_diff = torch.abs(rgb_feat - ir_feat)
        
        concat_feat = torch.cat([f_common, f_diff], dim=1)
        gap_feat = self.gap(concat_feat).flatten(1)
        competition_w = self.fc_common(gap_feat).view(-1, self.channels, 1, 1)
        
        enhanced_common = f_common + f_diff * competition_w
        
        output = self.enhance(enhanced_common) + enhanced_common
        
        noise_w = self.noise_gate(output).view(-1, self.channels, 1, 1)
        output = output * noise_w
        
        output = self.proj(output)
        output = self.bn_out(output)
        
        return output


class DifferentialFeatureEnhancementV2(nn.Module):
    """
    DFE V2: 带空间注意力增强的差分特征增强
    
    相比V1的改进:
    - 添加空间维度的门控机制 (像素级自适应)
    - 添加残差连接保证梯度流动
    - 更精细的特征选择机制
    
    Args:
        c1: 输入通道数
        c2: 输出通道数 (可选)
        reduction: 通道降维比例
    """
    def __init__(self, c1, c2=None, reduction=16):
        super().__init__()
        
        channels = c1[0] if isinstance(c1, list) else c1
        out_channels = c2 if c2 is not None else channels
        
        reduction_c = max(1, channels // reduction)
        self.channels = channels
        self.out_channels = out_channels
        
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 2, reduction_c),
            nn.GELU(),
            nn.Linear(reduction_c, channels * 2),
            nn.Sigmoid()
        )
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 7, padding=3, groups=channels // 4),
            nn.BatchNorm2d(channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if channels != out_channels else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        rgb_feat = x[0]
        ir_feat = x[1]
        
        f_common = (rgb_feat + ir_feat) / 2.0
        f_diff = torch.abs(rgb_feat - ir_feat)
        
        concat_feat = torch.cat([f_common, f_diff], dim=1)
        
        B, _, H, W = concat_feat.shape
        gap_vec = F.adaptive_avg_pool2d(concat_feat, 1).flatten(1)
        channel_w = self.channel_gate(gap_vec).view(B, self.channels * 2, 1, 1)
        w_common_ch, w_diff_ch = channel_w[:, :self.channels], channel_w[:, self.channels:]
        
        spatial_w = self.spatial_conv(concat_feat)
        w_common_sp, w_diff_sp = spatial_w[:, 0:1], spatial_w[:, 1:2]
        
        fused = f_common * w_common_ch * w_common_sp + f_diff * w_diff_ch * w_diff_sp
        
        enhanced = self.enhance_conv(fused)
        output = F.gelu(enhanced + fused)
        
        return self.output_proj(output)


if __name__ == "__main__":
    print("=" * 60)
    print("测试 DifferentialFeatureEnhancement (DFE)")
    print("=" * 60)
    
    batch_size = 2
    channels = 256
    height, width = 40, 40
    
    print(f"\n配置:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Channels: {channels}")
    print(f"  Feature Map Size: {height}x{width}")
    
    dfe = DifferentialFeatureEnhancement(c1=channels)
    total_params = sum(p.numel() for p in dfe.parameters())
    trainable_params = sum(p.numel() for p in dfe.parameters() if p.requires_grad)
    
    print(f"\n模块统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    rgb_feat = torch.randn(batch_size, channels, height, width)
    ir_feat = torch.randn(batch_size, channels, height, width)
    
    print(f"\n输入:")
    print(f"  RGB Feature: {rgb_feat.shape}")
    print(f"  IR Feature:  {ir_feat.shape}")
    
    output = dfe([rgb_feat, ir_feat])
    
    print(f"\n输出:")
    print(f"  Output Shape: {output.shape}")
    
    assert output.shape == (batch_size, channels, height, width), \
        f"输出维度不匹配! 期望 ({batch_size}, {channels}, {height}, {width}), 得到 {output.shape}"
    
    print("\n✅ 测试通过! 维度正确")
    
    print("\n" + "=" * 60)
    print("测试 DifferentialFeatureEnhancementV2 (DFE-V2)")
    print("=" * 60)
    
    dfe_v2 = DifferentialFeatureEnhancementV2(c1=channels)
    total_params_v2 = sum(p.numel() for p in dfe_v2.parameters())
    
    print(f"\nDFE-V2 参数量: {total_params_v2:,} (对比 DFE: {total_params:,})")
    
    output_v2 = dfe_v2([rgb_feat, ir_feat])
    assert output_v2.shape == (batch_size, channels, height, width)
    
    print("\n✅ DFE-V2 测试通过!")
    
    print("\n" + "=" * 60)
    print("计算复杂度分析 (FLOPs估算)")
    print("=" * 60)
    
    from thop import profile
    
    dummy_rgb = torch.randn(1, channels, height, width)
    dummy_ir = torch.randn(1, channels, height, width)
    
    flops_dfe, params_dfe = profile(dfe, inputs=([dummy_rgb, dummy_ir],), verbose=False)
    flops_v2, params_v2 = profile(dfe_v2, inputs=([dummy_rgb, dummy_ir],), verbose=False)
    
    print(f"\nDFE:")
    print(f"  FLOPs: {flops_dfe / 1e6:.2f} M")
    print(f"  Params: {params_dfe / 1e3:.2f} K")
    
    print(f"\nDFE-V2:")
    print(f"  FLOPs: {flops_v2 / 1e6:.2f} M")
    print(f"  Params: {params_v2 / 1e3:.2f} K")
    
    print(f"\n相对开销:")
    print(f"  DFE-V2/DFE FLOPs比: {flops_v2/flops_dfe:.2f}x")
    print(f"  DFE-V2/DFE Params比: {params_v2/params_dfe:.2f}x")