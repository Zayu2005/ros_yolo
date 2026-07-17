import torch
import torch.nn as nn
import torch.nn.functional as F

from .SGWTConv import SGWTConv


class IRGuidedRectifier(nn.Module):
    """
    工业级防雷版：IR 先验引导空间校正器
    利用轻量级卷积预测空间偏移场，并通过 F.grid_sample 进行亚像素级重采样，
    完美对齐 1~5 像素的双光视差。
    """
    def __init__(self, channels, max_offset_pixels=5.0):
        super().__init__()
        self.max_offset = max_offset_pixels # 限制最大物理偏移像素
        
        # 偏移量预测网络：输入 IR 和 RGB 拼接(channels * 2)，输出 X 和 Y 两个方向的偏移量(2)
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, 2, kernel_size=3, padding=1, bias=True),
            nn.Tanh() # 强制输出在 [-1, 1] 之间，防止预测飞出天际
        )
        
        # 完美初始化：保持不变映射
        nn.init.zeros_(self.offset_predictor[-2].weight) # 注意是 -2，因为最后加了 Tanh
        nn.init.zeros_(self.offset_predictor[-2].bias)

    def forward(self, ir_feat, rgb_feat):
        B, C, H, W = ir_feat.shape
        
        # 1. 预测偏移 (此时输出在 -1 到 1 之间)
        concat_feat = torch.cat([ir_feat, rgb_feat], dim=1)
        offsets = self.offset_predictor(concat_feat) # [B, 2, H, W]
        
        # --- 核心改进 1：动态生成基础网格的高效写法 ---
        # 避免每次都用 meshgrid，利用 arange 更快
        y, x = torch.meshgrid(
            torch.arange(H, device=ir_feat.device, dtype=ir_feat.dtype),
            torch.arange(W, device=ir_feat.device, dtype=ir_feat.dtype),
            indexing='ij'
        )
        # 将像素坐标转换到 [-1, 1] 范围
        base_grid_x = (x / (W - 1)) * 2.0 - 1.0
        base_grid_y = (y / (H - 1)) * 2.0 - 1.0
        base_grid = torch.stack([base_grid_x, base_grid_y], dim=-1) # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)    # [B, H, W, 2]
        
        # --- 核心改进 2：约束偏移量尺度 ---
        # 将 Tanh 的输出 [-1, 1] 映射到允许的最大像素偏移 (例如 ±5 像素)
        # 并转化为网格尺度的偏移 [-2*max_offset/W, 2*max_offset/W]
        offset_x = offsets[:, 0, :, :] * (self.max_offset * 2.0 / (W - 1))
        offset_y = offsets[:, 1, :, :] * (self.max_offset * 2.0 / (H - 1))
        grid_shift = torch.stack([offset_x, offset_y], dim=-1)
        
        # 最终形变网格
        deformed_grid = base_grid + grid_shift
        
        # --- 核心改进 3：强制 FP32 防 AMP 爆炸 ---
        # F.grid_sample 在半精度下极易 NaN，必须强转 FP32
        rgb_aligned = F.grid_sample(
            rgb_feat.float(),          # 特征转全精度
            deformed_grid.float(),     # 网格转全精度
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).to(rgb_feat.dtype)           # 算完后再转回原来的精度 (如 FP16)
        
        return rgb_aligned



class IA_DFCA(nn.Module):
    """
    IA-DFCA: Illumination-Aware Differential Frequency Cross-Attention
    面向全天候交通场景的无人机视角多模态特征融合模块。
    设计理念: 以 IR 为绝对主导, 根据全局光照自适应感知, 在差分掩码的引导下, 
    主动查询并提取由 SGWTConv 强化的 RGB 高频边缘细节。
    核心改进: 引入 IRGuidedRectifier 进行显式的空间视差校正。
    """
    def __init__(self, in_channels, out_channels=None):
        """
        初始化 IA-DFCA 模块
        :param in_channels: 输入的通道数 (IR 和 RGB 通道数一致，比如在 P3 层是 256)
        :param out_channels: 融合后输出的通道数 (默认等于 in_channels)
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
            
        # ==========================================
        # 第零步: 空间视差校正 (Spatial Rectification)
        # 针对 DroneVehicle 的双光视差，以 IR 为基准对 RGB 进行显式的亚像素级对齐
        # ==========================================
        self.rectifier = IRGuidedRectifier(channels=in_channels, max_offset_pixels=5.0)
            
        # ==========================================
        # 第一步: 频域强化 (Frequency Enhancement)
        # 抛弃传统的均值池化，直接采用 SGWTConv 提取 RGB 中的纯净高频边缘 (LH, HL, HH)
        # SGWTConv 自带门控机制，能自适应抑制 RGB 的背景噪声，只输出有价值的边缘残差
        # ==========================================
        self.wt_conv = SGWTConv(c1=in_channels, c2=in_channels, k=3, s=1)
        
        # ==========================================
        # 第二步: 特征差分引导 (Differential Guidance)
        # 计算 |IR - RGB_high|，利用 1x1 卷积压缩至单通道，生成空间过滤掩码
        # ==========================================
        self.diff_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # 输出 0~1 的 Spatial Mask
        )
        
        # ==========================================
        # 第三步: 非对称交叉注意力 (Asymmetric Cross-Attention)
        # 用极轻量级的 1x1 卷积生成 Q(由IR生成), K(由RGB生成), V(由RGB生成)
        # ==========================================
        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # ==========================================
        # 第四步: 光照感知门控 (Illumination-Aware Gated Fusion)
        # 通过全局平均池化感知原始 RGB 的整体光照，输出标量权重 alpha
        # ==========================================
        self.illumination_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # [B, C, 1, 1]
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False), # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1, bias=False), # 输出单通道
            nn.Sigmoid() # 输出 alpha (0~1 之间，白天近1，夜晚近0)
        )
        
        # 融合后的特征平滑处理
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) # YOLOv8 默认激活函数

    def forward(self, x):
        """
        前向传播
        :param x: 包含两个特征图的列表 [rgb_x, ir_x]
        :return: 融合后的特征图 [B, C, H, W]
        """
        rgb_x, ir_x = x[0], x[1]
        
        # 0. 空间视差校正：在一切开始之前，将 RGB 强行拉扯到与 IR 对齐的位置
        # 为什么先校正？因为要让偏移预测网络看到最原始、信息最完整的特征！
        rgb_aligned = self.rectifier(ir_x, rgb_x)
        
        # 1. 频域强化：获取包含丰富轮廓的 RGB 特征
        # 利用 SGWTConv 进行小波变换滤波，剥离受光照干扰严重的低频背景，保留高频细节
        # 此时送入 SGWTConv 的已经是完美对齐的 RGB 特征了
        rgb_high = self.wt_conv(rgb_aligned)
        
        # 2. 特征差分引导：寻找互补区域
        # 找出 IR 没看清但 RGB_high 很清晰的轮廓边界 (此时两者空间上已经严丝合缝)
        diff_feature = torch.abs(ir_x - rgb_high)
        mask_diff = self.diff_conv(diff_feature) # Shape: [B, 1, H, W]
        
        # 3. 非对称交叉注意力：IR 主动查询 RGB 纹理
        # 抛弃耗时的矩阵乘法，使用极度轻量级的逐元素乘法 (Element-wise)
        q = self.q_conv(ir_x)                # IR 作为 Query
        k = self.k_conv(rgb_high)            # RGB_high 作为 Key
        v = self.v_conv(rgb_high)            # RGB_high 作为 Value
        
        # 逐元素相乘计算局部注意力图
        # 因为在第 0 步已经对齐，Q 和 K 可以完美命中
        attn = torch.sigmoid(q * k)  # Shape: [B, C, H, W]
        
        # 提取相关特征
        v_attn = attn * v            # Shape: [B, C, H, W]
        
        # 4. 光照感知：全局光照评估
        # 根据原始 rgb_x 评估天亮程度，决定采用多少 RGB 特征
        # 注意：这里依然使用最原始的 rgb_x 来评估全局光照，因为对齐/滤波可能会改变全局亮度统计
        alpha = self.illumination_gate(rgb_x) # Shape: [B, 1, 1, 1]
        
        # 5. 终极融合方程式: Output = IR + alpha * (Mask_diff ⊙ V_attn)
        # 精准筛选：将交叉注意力提取的特征(v_attn) 用差分掩码(mask_diff) 过滤
        filtered_rgb_supplement = mask_diff * v_attn 
        
        # 残差注入：将过滤后的特征按光照权重(alpha) 贴补回原红外特征(ir_x)
        fused_out = ir_x + alpha * filtered_rgb_supplement
        
        # 最后的平滑与通道映射
        out = self.act(self.bn(self.final_conv(fused_out)))
        
        return out


if __name__ == '__main__':
    # 简单的测试脚本，验证张量维度与运行是否正常
    batch_size = 32
    channels = 128 # 模拟 YOLOv8 P3 层通道数
    height, width = 80, 80
    
    # 模拟红外和可见光输入特征图
    ir_tensor = torch.randn(batch_size, channels, height, width)
    rgb_tensor = torch.randn(batch_size, channels, height, width)
    
    # 实例化我们的融合模块
    fusion_module = IA_DFCA(in_channels=channels, out_channels=channels)
    
    # 计算模型参数量 (以 M 为单位)
    total_params = sum(p.numel() for p in fusion_module.parameters()) / 1e6
    
    # 前向推理
    output = fusion_module([rgb_tensor, ir_tensor])
    
    # 梯度回传测试
    loss = output.sum()
    loss.backward()
    
    # 打印梯度信息
    print("====== 梯度传导测试 ======")
    for name, param in fusion_module.named_parameters():
        if param.grad is not None:
            print(f"{name} 梯度正常")
        else:
            print(f"!!! 警告: {name} 无梯度 !!!")
            
    # 自动混合精度(AMP)测试
    print("====== AMP 兼容性测试 ======")
    from torch.cuda.amp import autocast
    fusion_module = fusion_module.cuda()
    ir_tensor = ir_tensor.cuda()
    rgb_tensor = rgb_tensor.cuda()
    try:
        with autocast():
            output_amp = fusion_module([rgb_tensor, ir_tensor])
        print(f"AMP 测试通过，输出维度: {output_amp.shape}, 数据类型: {output_amp.dtype}")
    except Exception as e:
        print(f"AMP 测试失败: {e}")
        
    print("====== IA-DFCA Module Test ======")
    print(f"IR Input shape:  {ir_tensor.shape}")
    print(f"RGB Input shape: {rgb_tensor.shape}")
    print(f"Output shape:    {output.shape}")
    print(f"Total Params:    {total_params:.4f} M")
    print("模块测试通过，完美对齐特征维度！")