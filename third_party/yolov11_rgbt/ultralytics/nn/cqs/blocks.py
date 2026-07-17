# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CrossModalAttention(nn.Module):
#     """轻量跨模态注意力：用模态A的特征引导模态B的特征重标定"""
#     def __init__(self, in_channels, reduction=4):
#         super().__init__()
#         # 通道注意力生成分支
#         self.channel_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels // reduction, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction, in_channels, 1),
#             nn.Sigmoid()
#         )
#         # 空间注意力生成分支
#         self.spatial_att = nn.Sequential(
#             nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         """
#         Args:
#             src (Tensor): 待增强特征 [B, C, H, W]
#             guide (Tensor): 引导特征   [B, C, H, W]
#         Returns:
#             enhanced (Tensor): 增强后特征 [B, C, H, W]
#         """
#         src, guide = x[0], x[1]
#         # 通道注意力由 guide 产生，作用到 src
#         ch_att = self.channel_att(guide)
#         src = src * ch_att

#         # 空间注意力由 guide 产生，作用到 src
#         sp_att = self.spatial_att(guide)
#         src = src * sp_att

#         # 进行残差连接
#         return x[0] + src

# class AdaptiveGateFusion(nn.Module):
#     """自适应门控融合：学习两个模态的保留权重"""
#     def __init__(self, in_channels):
#         super().__init__()
#         self.gate_conv = nn.Sequential(
#             nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, 2, 3, padding=1),  # 输出两个通道的权重图
#             nn.Sigmoid()
#         )

#     def forward(self, feat_a, feat_b):
#         """
#         Args:
#             feat_a: 模态A原始特征
#             feat_b: 模态B原始特征
#         Returns:
#             w_a, w_b: 权重图 [B,1,H,W]
#         """
#         concat = torch.cat([feat_a, feat_b], dim=1)
#         weights = self.gate_conv(concat)        # [B,2,H,W]
#         w_a = weights[:, 0:1, :, :]
#         w_b = weights[:, 1:2, :, :]
#         return w_a, w_b

# class DualModalInteractionBlock(nn.Module):
#     """双模态交互融合模块（插入 backbone 各层）"""
#     def __init__(self, in_channels, reduction=8, residual_weight=0.5):
#         super().__init__()
#         self.residual_weight = residual_weight

#         # RGB 引导 IR 增强
#         self.rgb_to_ir = CrossModalAttention(in_channels, reduction)
#         # IR 引导 RGB 增强
#         self.ir_to_rgb = CrossModalAttention(in_channels, reduction)

#         # 自适应门控
#         self.gate = AdaptiveGateFusion(in_channels)

#         # 融合后可选的特征平滑卷积
#         self.smooth_rgb = nn.Conv2d(in_channels, in_channels, 3, padding=1)
#         self.smooth_ir  = nn.Conv2d(in_channels, in_channels, 3, padding=1)

#     def forward(self, x):
#         """
#         Args:
#             feat_rgb: RGB 模态特征 [B, C, H, W]
#             feat_ir:  IR 模态特征  [B, C, H, W]
#         Returns:
#             out_rgb: 增强后的 RGB 特征
#             out_ir:  增强后的 IR 特征
#         """
#         feat_rgb, feat_ir = x[0], x[1]
#         # 1. 跨模态增强
#         enhanced_rgb = self.ir_to_rgb((feat_rgb, feat_ir))   # IR引导RGB
#         enhanced_ir  = self.rgb_to_ir((feat_ir, feat_rgb))   # RGB引导IR

#         # 2. 自适应门控权重
#         w_rgb, w_ir = self.gate(feat_rgb, feat_ir)

#         # 3. 加权融合原始特征与增强特征
#         fused_rgb = w_rgb * feat_rgb + (1 - w_rgb) * enhanced_rgb
#         fused_ir  = w_ir  * feat_ir  + (1 - w_ir)  * enhanced_ir

#         # 4. 平滑与残差连接（稳定训练，保留原始信息）
#         out_rgb = feat_rgb + self.residual_weight * self.smooth_rgb(fused_rgb)
#         out_ir  = feat_ir  + self.residual_weight * self.smooth_ir(fused_ir)

#         return [out_rgb, out_ir]

# if __name__ == '__main__':
#     # 假设两个模态的特征形状均为 [B, 64, 32, 32]
#     src_feat = torch.randn(4, 64, 32, 32)
#     guide_feat = torch.randn(4, 64, 32, 32)

#     model = CrossModalAttention(in_channels=64, reduction=8)
#     output = model((src_feat, guide_feat))   # 当前接口
#     print(output.shape)
#     # 建议改为 output = model(src_feat, guide_feat)


import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C2f


class FeatureAlignment(nn.Module):
    """轻量特征对齐：预测偏移量，对 guide 进行 warp 使其对齐 src"""
    def __init__(self, in_channels):
        super().__init__()
        # 用两个模态拼接预测 2 通道偏移场 (dx, dy)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, 3, padding=1)   # 输出 offset map
        )
        # 初始化最后一层接近0，使初始 warp 接近恒等变换
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)

    def forward(self, src, guide):
        """
        Args:
            src:   目标模态特征 [B, C, H, W] （对齐参考）
            guide: 待对齐特征   [B, C, H, W]
        Returns:
            aligned_guide: 对齐后的 guide 特征
        """
        concat = torch.cat([src, guide], dim=1)
        offset = self.offset_conv(concat)   # [B, 2, H, W]

        # 生成采样网格（归一化坐标）
        B, _, H, W = offset.shape
        yy, xx = torch.meshgrid(
            torch.arange(H, device=offset.device, dtype=offset.dtype),
            torch.arange(W, device=offset.device, dtype=offset.dtype),
            indexing='ij'
        )
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # [B,2,H,W]

        # 偏移量归一化到 [-1, 1] 范围
        norm_offset = offset.clone()
        norm_offset[:, 0] = offset[:, 0] / (W - 1) * 2.0   # x方向
        norm_offset[:, 1] = offset[:, 1] / (H - 1) * 2.0   # y方向

        sampling_grid = (grid + norm_offset).permute(0, 2, 3, 1)  # [B,H,W,2]

        aligned_guide = F.grid_sample(guide, sampling_grid, align_corners=True, padding_mode='border')
        return aligned_guide


class CrossModalAttention(nn.Module):
    """跨模态注意力：用对齐后的 guide 重标定 src"""
    def __init__(self, in_channels, reduction=4, use_alignment=True):
        super().__init__()
        self.use_alignment = use_alignment
        if use_alignment:
            self.align = FeatureAlignment(in_channels)

        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, src, guide):
        """
        Args:
            src (Tensor): 待增强特征 [B, C, H, W]
            guide (Tensor): 引导特征   [B, C, H, W]
        Returns:
            enhanced (Tensor): 增强后特征 [B, C, H, W]
        """
        # 1. 特征对齐（可选）
        if self.use_alignment:
            guide = self.align(src, guide)

        # 2. 通道注意力
        ch_att = self.channel_att(guide)
        src = src * ch_att

        # 3. 空间注意力
        sp_att = self.spatial_att(guide)
        src = src * sp_att

        return src


class AdaptiveGateFusion(nn.Module):
    """自适应门控融合：学习两个模态的保留权重"""
    def __init__(self, in_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, feat_a, feat_b):
        concat = torch.cat([feat_a, feat_b], dim=1)
        weights = self.gate_conv(concat)        # [B,2,H,W]
        w_a = weights[:, 0:1, :, :]
        w_b = weights[:, 1:2, :, :]
        return w_a, w_b


class DualModalInteractionBlock(nn.Module):
    """双模态交互融合模块（支持不对齐特征）"""
    def __init__(self, in_channels, reduction=8, residual_weight=0.5, use_alignment=True):
        super().__init__()
        self.residual_weight = residual_weight

        # 跨模态增强（可开启对齐）
        self.rgb_to_ir = CrossModalAttention(in_channels, reduction, use_alignment)
        self.ir_to_rgb = CrossModalAttention(in_channels, reduction, use_alignment)

        # 自适应门控
        self.gate = AdaptiveGateFusion(in_channels)

        # 融合后平滑
        self.smooth_rgb = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.smooth_ir  = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        """
        Args:
            feat_rgb: RGB 特征 [B, C, H, W]
            feat_ir:  IR 特征  [B, C, H, W]
        Returns:
            out_rgb, out_ir
        """
        feat_rgb, feat_ir = x[0], x[1]
        # 1. 跨模态增强
        enhanced_rgb = self.ir_to_rgb(feat_rgb, feat_ir)   # IR引导RGB
        enhanced_ir  = self.rgb_to_ir(feat_ir, feat_rgb)   # RGB引导IR

        # 2. 门控权重
        w_rgb, w_ir = self.gate(feat_rgb, feat_ir)

        # 3. 加权融合
        fused_rgb = w_rgb * feat_rgb + (1 - w_rgb) * enhanced_rgb
        fused_ir  = w_ir  * feat_ir  + (1 - w_ir)  * enhanced_ir

        # 4. 平滑 + 残差
        out_rgb = feat_rgb + self.residual_weight * self.smooth_rgb(fused_rgb)
        out_ir  = feat_ir  + self.residual_weight * self.smooth_ir(fused_ir)

        return [out_rgb, out_ir]

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))




class TextureEnhance(nn.Module):
    """
    轻量纹理增强模块，在特征图上提取可学习的边缘/纹理特征。
    内部使用可学习的 Sobel 风格卷积核（初始化后允许微调），输出指定通道数的纹理特征图。
    """
    def __init__(self, out_channels, edge_channels=16):
        """
        Args:
            out_channels: 输出特征图的通道数（用于后续 concat / add）
            edge_channels: 边缘响应图的通道数，默认 16
        """
        super().__init__()
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self._edge_conv = None      # 延迟创建，需要知道输入通道数
        self._reduce_conv = None

    def forward(self, x):
        # 首次 forward 时动态创建卷积层（根据输入通道数）
        if self._edge_conv is None:
            in_channels = x.shape[1]
            # 边缘提取卷积：3x3，无 bias
            self._edge_conv = nn.Conv2d(in_channels, self.edge_channels, 3, padding=1, bias=False)
            self._init_sobel_weights(self._edge_conv, in_channels, self.edge_channels)
            self._edge_conv = self._edge_conv.to(x.device)
            # 通道变换：1x1 卷积
            self._reduce_conv = nn.Conv2d(self.edge_channels, self.out_channels, 1)
            self._reduce_conv = self._reduce_conv.to(x.device)

        # 计算边缘响应 + ReLU 激活
        edge_feat = self._edge_conv(x)
        edge_feat = F.relu(edge_feat)
        # 调整通道数
        out = self._reduce_conv(edge_feat)
        return out

    def _init_sobel_weights(self, conv, in_c, out_c):
        """用 Sobel 算子初始化卷积核的前两个输出通道，其余随机"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # 可选归一化：使响应范围与常规特征相近
        sobel_x /= 4.0
        sobel_y /= 4.0

        with torch.no_grad():
            conv.weight.zero_()
            for i in range(min(out_c, 2)):
                kernel = sobel_x if i == 0 else sobel_y
                # 对所有输入通道使用相同的核（相当于各通道边缘响应求和）
                for j in range(in_c):
                    conv.weight[i, j] = kernel
            # 其余输出通道随机初始化
            for i in range(2, out_c):
                nn.init.normal_(conv.weight[i], 0, 0.01)


class C2f_TextureEnhance(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, texture_out=64):
        super().__init__()
        self.c2f = C2f(c1, c2, n, shortcut, g, e)
        self.texture = TextureEnhance(out_channels=texture_out)
        self.reduce = nn.Conv2d(c2 + texture_out, c2, 1)

    def forward(self, x):
        feat_c2f = self.c2f(x)
        feat_texture = self.texture(x)
        out = self.reduce(torch.cat([feat_c2f, feat_texture], dim=1))
        return out

import math

class FreqEnhance(nn.Module):
    """
    频率增强模块：专为双模态(RGB+IR)设计
    YAML 用法: [[10, 20], 1, FreqEnhance, [1024]]  # c2=1024
    """
    def __init__(self, c1, c2, k=3, reduction=8):
        super().__init__()
        # c1: 单个模态输入通道数（框架自动从列表第一个元素推断）
        # c2: 期望的输出通道数（来自 YAML args[0]）
        self.c1 = c1
        self.c2 = c2
        
        # 高频增强通道注意力（SE-Block）
        self.se_fc1 = nn.Conv2d(c1, c1 // reduction, 1)
        self.se_fc2 = nn.Conv2d(c1 // reduction, c1, 1)
        
        # 低频平滑卷积（深度可分离，减少计算量）
        self.low_smooth = nn.Conv2d(c1, c1, 1, groups=c1, bias=False)
        
        # 关键修正：拼接后投影到目标通道数 c2
        self.proj = nn.Conv2d(c1 * 2, c2, 1) if c1 * 2 != c2 else nn.Identity()

    def forward(self, x):
        if isinstance(x, list):
            feat_rgb, feat_ir = x[0], x[1]
        else:
            # 防御性代码：若已拼接，按通道切分
            c_split = x.shape[1] // 2
            feat_rgb, feat_ir = x[:, :c_split], x[:, c_split:]

        enhanced_feats = []
        for feat in (feat_rgb, feat_ir):
            low_freq = F.adaptive_avg_pool2d(feat, 1)
            high_freq = feat - low_freq
            
            se_weight = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(low_freq))))
            high_freq_enhanced = high_freq * se_weight
            
            low_freq_smooth = self.low_smooth(low_freq.expand_as(feat))
            
            enhanced = low_freq_smooth + high_freq_enhanced
            enhanced_feats.append(enhanced)
        
        # 拼接并投影到目标通道数
        out = torch.cat(enhanced_feats, dim=1)   # (B, 2*c1, H, W)
        return self.proj(out)                    # (B, c2, H, W)

class EdgeGate(nn.Module):
    """
    形状增强模块：利用浅层 RGB 图像的真实 Sobel 边缘，作为深层特征的空间注意力掩膜。
    输入格式: [raw_rgb_feat, target_feat] 
        - raw_rgb_feat: 来自主干早期层（如第1层），包含未下采样太多的 RGB 信息
        - target_feat: 需要被增强的深层特征图（如第6层 P3）
    """
    def __init__(self, c1, c2, k=3):
        super().__init__()
        # c1 是 target_feat 的通道数
        self.c_in = c1
        self.c_out = c2
        # 可训练的缩放因子，决定边缘先验的介入强度
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # Sobel 算子 Kernel (固定)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # 将单通道边缘映射到 target_feat 的通道空间
        self.edge_project = nn.Conv2d(1, c1, 1, bias=False)
        # 可选的融合层（若输入输出通道数不同）
        self.fuse = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def extract_edge(self, x_rgb):
        """
        从 RGB 特征图计算 Sobel 边缘强度。
        假设 x_rgb 的形状为 (N, 3, H, W) 且值域在 ImageNet 归一化范围或 0-1 之间。
        """
        # 如果是 4 通道（RGB+IR），仅取前 3 通道
        if x_rgb.shape[1] == 4:
            x_rgb = x_rgb[:, :3, :, :]
        elif x_rgb.shape[1] != 3:
            # 若不是标准 RGB，取均值模拟灰度（但这种情况不应出现，我们已从第1层正确传入）
            x_rgb = x_rgb.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        
        # 转为灰度：标准 RGB 权重
        gray = 0.299 * x_rgb[:, 0:1, :, :] + 0.587 * x_rgb[:, 1:2, :, :] + 0.114 * x_rgb[:, 2:3, :, :]
        
        # Sobel 边缘检测
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(gx**2 + gy**2 + 1e-6)
        
        # 归一化到 [0, 1]
        edge = edge / (edge.max() + 1e-8)
        return edge

    def forward(self, inputs):
        """
        inputs: list 包含 [raw_rgb_feat, target_feat]
        """
        # return inputs[1]
        raw_feat, target_feat = inputs[0], inputs[1]
        
        # 计算边缘图
        edge_map = self.extract_edge(raw_feat)
        
        # 若边缘图尺寸与 target_feat 不同，进行下采样
        if edge_map.shape[-2:] != target_feat.shape[-2:]:
            edge_map = F.interpolate(edge_map, size=target_feat.shape[-2:], mode='bilinear', align_corners=False)
        
        # 投影到特征通道空间
        edge_feat = self.edge_project(edge_map)
        
        # 生成门控掩膜
        gate = torch.sigmoid(edge_feat)
        
        # 形状增强：原始特征 + alpha * (原始特征 * 边缘门控)
        enhanced = target_feat + self.alpha * (target_feat * gate)
        
        return self.fuse(enhanced)
