import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGuidedMultiModalAttention(nn.Module):
    """
    光照引导的多模态注意力融合 (替换主干 Concat)
    双输入 [rgb_feat, ir_feat]，单输出
    """
    def __init__(self, c1, c2=None, reduction=16):
        super().__init__()
        # 兼容 YOLO 的通道解析
        channels = c1[0] if isinstance(c1, list) else c1
        out_channels = c2 if c2 is not None else channels
        
        reduction_c = max(1, channels // reduction)
        
        # 光照条件编码（全局）
        self.global_light_encoder = nn.Sequential(
            nn.Linear(channels * 2, reduction_c),
            nn.ReLU(),
            nn.Linear(reduction_c, channels * 2)
        )
        
        # 空间注意力：输入包含 拼接特征(2C) + MaxPool(1) + AvgPool(1) = 2C + 2
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 2, 1),  # 2通道：RGB权重和IR权重
            nn.Sigmoid()
        )
        
        # 维度对齐层
        self.proj = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()
        
    def forward(self, x):
        """
        x: list 包含两个特征图 [rgb_feat, ir_feat]
        """
        rgb_feat, ir_feat = x[0], x[1]
        B, C, H, W = rgb_feat.shape
        
        # 1. 全局光照影响通道注意力
        cat_pool = torch.cat([
            F.adaptive_avg_pool2d(rgb_feat, 1).flatten(1),
            F.adaptive_avg_pool2d(ir_feat, 1).flatten(1)
        ], dim=1) # [B, 2C]
        
        channel_bias = self.global_light_encoder(cat_pool).view(B, C * 2, 1, 1)
        channel_bias_rgb, channel_bias_ir = channel_bias.chunk(2, dim=1)
        
        # 2. 逐元素加权（通道偏置）
        rgb_mod = rgb_feat * (1 + torch.tanh(channel_bias_rgb))
        ir_mod = ir_feat * (1 + torch.tanh(channel_bias_ir))
        
        # 3. 空间注意力生成融合权重
        spatial_input_base = torch.cat([rgb_mod, ir_mod], dim=1) # [B, 2C, H, W]
        
        # 补充空间维度的 MaxPool 和 AvgPool 特征 (类似于 CBAM 的空间注意力设计)
        max_pool = torch.max(spatial_input_base, dim=1, keepdim=True)[0] # [B, 1, H, W]
        avg_pool = torch.mean(spatial_input_base, dim=1, keepdim=True)   # [B, 1, H, W]
        
        spatial_input = torch.cat([spatial_input_base, max_pool, avg_pool], dim=1) # [B, 2C+2, H, W]
        
        fusion_weights = self.spatial_attn(spatial_input)  # [B, 2, H, W]
        w_rgb, w_ir = fusion_weights[:, 0:1], fusion_weights[:, 1:2]
        
        # 归一化权重
        total = w_rgb + w_ir + 1e-8
        w_rgb = w_rgb / total
        w_ir = w_ir / total
        
        fused = w_rgb * rgb_feat + w_ir * ir_feat
        
        # 返回对齐通道后的单张量
        return self.proj(fused)

if __name__ == "__main__":
    # 测试代码
    # 假设输入特征通道数为 64，YOLO 默认 Concat 输出通道为 128 (64*2)
    in_channels = 64
    out_channels = 128
    batch_size = 2
    height, width = 32, 32
    
    # 初始化模块
    model = LightGuidedMultiModalAttention(c1=in_channels, c2=out_channels)
    print(f"初始化模型 LightGuidedMultiModalAttention: in_channels={in_channels}, out_channels={out_channels}")
    
    # 模拟输入特征 (RGB 和 IR)
    rgb_feature = torch.randn(batch_size, in_channels, height, width)
    ir_feature = torch.randn(batch_size, in_channels, height, width)
    
    # YOLO 中 Concat 层的输入是一个包含多个特征图的列表
    input_features = [rgb_feature, ir_feature]
    
    # 前向传播
    output = model(input_features)
    
    # 验证输出维度
    print(f"输入特征维度 (单模态): {rgb_feature.shape}")
    print(f"输出特征维度: {output.shape}")
    assert output.shape == (batch_size, out_channels, height, width), "输出维度不匹配！"
    print("测试通过！输出维度正确。")
