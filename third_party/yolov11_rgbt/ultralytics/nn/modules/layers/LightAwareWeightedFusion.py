import torch
import torch.nn as nn
import torch.nn.functional as F

class LightAwareWeightedFusion(nn.Module):
    """
    光照感知自适应加权融合 (替换主干 Concat)
    双输入 [rgb_feat, ir_feat]，单输出
    """
    def __init__(self, c1, c2=None, hidden_dim=64):
        super().__init__()
        # 兼容 YOLO 的通道解析，c1 可能是列表 [c_rgb, c_ir] 或单个整数
        channels = c1[0] if isinstance(c1, list) else c1
        out_channels = c2 if c2 is not None else channels
        
        # 光照感知编码器：从特征提取光照特征
        self.light_encoder = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出[0,1]权重，1=完全RGB，0=完全IR
        )
        
        # 维度对齐层
        self.proj = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()
        
    def forward(self, x):
        """
        x: list 包含两个特征图 [rgb_feat, ir_feat]
        """
        rgb_feat, ir_feat = x[0], x[1]
        
        # 从 RGB 特征的全局均值估计光照条件
        light_feat = F.adaptive_avg_pool2d(rgb_feat, 1).flatten(1) # [B, C]
        
        # 计算融合权重
        w = self.light_encoder(light_feat).view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # 加权融合
        fused = w * rgb_feat + (1 - w) * ir_feat
        
        return self.proj(fused)

if __name__ == "__main__":
    # 测试代码
    # 假设输入特征通道数为 64，YOLO 默认 Concat 输出通道为 128 (64*2)
    in_channels = 64
    out_channels = 128
    batch_size = 2
    height, width = 32, 32
    
    # 初始化模块
    model = LightAwareWeightedFusion(c1=in_channels, c2=out_channels)
    print(f"初始化模型 LightAwareWeightedFusion: in_channels={in_channels}, out_channels={out_channels}")
    
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
