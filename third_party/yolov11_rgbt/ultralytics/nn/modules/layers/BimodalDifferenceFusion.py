import torch
import torch.nn as nn
import torch.nn.functional as F

class BimodalDifferenceFusion(nn.Module):
    """
    双模态差分融合 (替换主干 Concat)
    双输入 [rgb_feat, ir_feat]，单输出
    """
    def __init__(self, c1, c2=None, reduction=8):
        super().__init__()
        # 兼容 YOLO 的通道解析，c1 可能是列表 [c_rgb, c_ir] 或单个整数
        channels = c1[0] if isinstance(c1, list) else c1
        out_channels = c2 if c2 is not None else channels
        
        # 差分注意力生成器
        reduction_c = max(1, channels // reduction)
        self.diff_attn = nn.Sequential(
            nn.Conv2d(channels, reduction_c, 1, bias=False),
            nn.BatchNorm2d(reduction_c),
            nn.ReLU(),
            nn.Conv2d(reduction_c, channels, 1, bias=False),
            nn.Tanh()  # [-1,1]范围
        )
        
        # 模态门控：判断哪个模态更可靠
        self.modal_gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Linear(channels, 2),
            nn.Softmax(dim=1)
        )
        
        # 维度对齐层 (如果替换Concat，往往需要输出2C通道以匹配后续网络)
        self.proj = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()
        
    def forward(self, x):
        """
        x: list 包含两个特征图 [rgb_feat, ir_feat]
        """
        rgb_feat, ir_feat = x[0], x[1]
        
        # 1. 差分特征
        diff = rgb_feat - ir_feat
        diff_weight = self.diff_attn(diff)  # [-1,1]权重
        
        # 2. 模态可靠性门控
        gate_input = torch.cat([
            F.adaptive_avg_pool2d(rgb_feat, 1).flatten(1),
            F.adaptive_avg_pool2d(ir_feat, 1).flatten(1)
        ], dim=1) # [B, 2C]
        
        gate = self.modal_gate(gate_input)  # [B, 2]
        
        # 3. 动态融合
        g_rgb, g_ir = gate[:, 0:1], gate[:, 1:2]
        diff_strength = g_ir - g_rgb  # [-1,1]
        diff_contribution = diff_strength.unsqueeze(-1).unsqueeze(-1) * diff_weight * diff
        
        fused = rgb_feat + ir_feat + diff_contribution
        
        # 维度对齐后返回单张量
        return self.proj(fused)

if __name__ == "__main__":
    # 测试代码
    # 假设输入特征通道数为 64，YOLO 默认 Concat 输出通道为 128 (64*2)
    in_channels = 64
    out_channels = 128
    batch_size = 2
    height, width = 32, 32
    
    # 初始化模块
    model = BimodalDifferenceFusion(c1=in_channels, c2=out_channels)
    print(f"初始化模型 BimodalDifferenceFusion: in_channels={in_channels}, out_channels={out_channels}")
    
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
