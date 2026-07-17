import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedPool2d(nn.Module):
    """
    SFS pooling version in the paper:
        F_a = AvgPool(F)
        F_m = MaxPool(F)
        F_o = lambda * F_a + (1-lambda) * F_m
    where lambda is learnable.
    """

    def __init__(self, kernel_size: int = 2, stride: Optional[int] = None, init_lambda: float = 0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        # use sigmoid(parameter) to keep the weight in [0, 1]
        init = torch.tensor(init_lambda).clamp(1e-4, 1 - 1e-4)
        self.logit_lambda = nn.Parameter(torch.log(init / (1 - init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fa = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        fm = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        lam = torch.sigmoid(self.logit_lambda)
        return lam * fa + (1.0 - lam) * fm


class SFSConv(nn.Module):
    """
    Spatial Feature Shrinking with convolution-style spatial-to-channel shrinking.
    A practical faithful implementation:
    use PixelUnshuffle to move spatial info into channel dimension,
    then compress channels with 1x1 conv.
    """

    def __init__(self, channels: int, scale: int = 2):
        super().__init__()
        assert scale >= 1 and isinstance(scale, int)
        self.scale = scale
        if scale == 1:
            self.op = nn.Identity()
        else:
            self.op = nn.Sequential(
                nn.PixelUnshuffle(scale),                  # B, C*s^2, H/s, W/s
                nn.Conv2d(channels * scale * scale, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SFS(nn.Module):
    """
    Spatial Feature Shrinking module.

    mode='pool'  -> mixed avg/max pooling (recommended by the paper)
    mode='conv'  -> reshape/spatial-to-channel + 1x1 conv approximation
    mode='none'  -> no shrinking
    """

    def __init__(self, channels: int, scale: int = 2, mode: str = "pool"):
        super().__init__()
        self.mode = mode
        if mode == "pool":
            self.op = MixedPool2d(kernel_size=scale, stride=scale)
        elif mode == "conv":
            self.op = SFSConv(channels, scale=scale)
        elif mode == "none" or scale == 1:
            self.op = nn.Identity()
        else:
            raise ValueError(f"Unsupported SFS mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossFeatureEnhancement(nn.Module):
    """
    CFE module from the paper.

    For enhancing target modality T using auxiliary modality R:
      - Q comes from auxiliary tokens
      - K,V come from target tokens
      - output keeps target token shape

    This exactly follows the paper's idea in Eq. (5)-(8), with:
      Z = softmax(Q_aux K_tgt^T / sqrt(d)) V_tgt
      T' = alpha * WO(Z) + beta * T
      T_hat = gamma * T' + delta * FFN(T')

    The paper uses learnable residual coefficients alpha/beta/gamma/delta.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.ffn = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)

        # learnable residual coefficients, initialized as 1
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C] -> [B, h, N, d]
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, h, N, d] -> [B, N, C]
        B, h, N, d = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, N, h * d)

    def forward(self, aux_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        aux_tokens: [B, N, C]  (e.g. RGB tokens used as queries)
        tgt_tokens: [B, N, C]  (e.g. Thermal tokens used as keys/values and output carrier)
        returns:    [B, N, C]  enhanced target tokens
        """
        q = self._reshape_heads(self.q_proj(aux_tokens))
        k = self._reshape_heads(self.k_proj(tgt_tokens))
        v = self._reshape_heads(self.v_proj(tgt_tokens))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        z = torch.matmul(attn, v)                          # [B, h, N, d]
        z = self._merge_heads(z)                           # [B, N, C]
        z = self.out_proj(z)

        t_prime = self.alpha * z + self.beta * tgt_tokens
        out = self.gamma * t_prime + self.delta * self.ffn(t_prime)
        return out


class ICFE(nn.Module):
    """
    Iterative Cross-modal Feature Enhancement.

    The same two CFE modules are reused across iterations (shared parameters),
    matching the paper's iterative-learning idea.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_iters: int = 1,
    ):
        super().__init__()
        self.num_iters = num_iters
        self.cfe_rgb = CrossFeatureEnhancement(dim, num_heads, mlp_ratio, dropout)
        self.cfe_t = CrossFeatureEnhancement(dim, num_heads, mlp_ratio, dropout)

    def forward(self, rgb_tokens: torch.Tensor, t_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r, t = rgb_tokens, t_tokens
        for _ in range(self.num_iters):
            # use old r,t for the current iteration, then update together
            new_r = self.cfe_rgb(aux_tokens=t, tgt_tokens=r)   # enhance RGB using Thermal as queries
            new_t = self.cfe_t(aux_tokens=r, tgt_tokens=t)     # enhance Thermal using RGB as queries
            r, t = new_r, new_t
        return r, t


class NINFusion(nn.Module):
    """1x1 conv fusion used by the paper as the local bimodal fusion head."""

    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([x1, x2], dim=1))


class DMFF(nn.Module):
    """
    Dual-modal Feature Fusion module from ICAFusion.

    Pipeline:
      1) SFS on RGB and Thermal feature maps
      2) flatten + positional embedding -> tokens
      3) ICFE with shared parameters across iterations
      4) reshape back to feature maps
      5) bilinear upsample to original spatial size
      6) residual add with original modality features
      7) NIN fusion to get final fused feature

    Inputs:
      rgb_feat: [B, C, H, W]
      t_feat:   [B, C, H, W]

    Outputs:
      fused:    [B, C, H, W]
      rgb_hat:  enhanced RGB branch after upsampling + residual add
      t_hat:    enhanced Thermal branch after upsampling + residual add
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        sfs_scale: int = 2,
        sfs_mode: str = "pool",
        num_iters: int = 1,
        max_tokens_hw: Optional[Tuple[int, int]] = None,
        use_pos_embed: bool = True,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        assert in_ch == out_ch, f"DMFF中in_ch:{in_ch} != out_ch:{out_ch}"
        channels = in_ch
        self.channels = channels
        self.use_pos_embed = use_pos_embed
        self.max_tokens_hw = max_tokens_hw

        self.sfs_rgb = SFS(channels, scale=sfs_scale, mode=sfs_mode)
        self.sfs_t = SFS(channels, scale=sfs_scale, mode=sfs_mode)
        self.icfe = ICFE(
            dim=channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_iters=num_iters,
        )
        self.fusion = NINFusion(channels, out_channels=out_channels or channels)

        # Optional learnable positional embedding. Since feature-map size may vary,
        # we store a base tensor and interpolate it when needed.
        if use_pos_embed:
            if max_tokens_hw is None:
                # sensible default for common P3/P4/P5 feature scales after SFS
                max_tokens_hw = (40, 40)
            self.base_pos = nn.Parameter(torch.zeros(1, channels, max_tokens_hw[0], max_tokens_hw[1]))
            nn.init.trunc_normal_(self.base_pos, std=0.02)
        else:
            self.register_parameter("base_pos", None)

    def _get_pos_embed(self, h: int, w: int) -> Optional[torch.Tensor]:
        if self.base_pos is None:
            return None
        pos = F.interpolate(self.base_pos, size=(h, w), mode="bilinear", align_corners=False)
        pos = pos.flatten(2).transpose(1, 2).contiguous()  # [1, N, C]
        return pos

    @staticmethod
    def _to_tokens(x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, H*W, C]
        return x.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _to_map(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # [B, H*W, C] -> [B, C, H, W]
        B, N, C = tokens.shape
        assert N == h * w, f"Token number {N} does not match h*w={h*w}"
        return tokens.transpose(1, 2).contiguous().view(B, C, h, w)

    def forward(self, x):
        rgb_feat, t_feat = x[0], x[1]
        assert rgb_feat.shape == t_feat.shape, "RGB/Thermal feature shapes must match"
        B, C, H, W = rgb_feat.shape
        assert C == self.channels, f"Expected channels={self.channels}, got {C}"

        # 1) shrink spatial size
        rgb_s = self.sfs_rgb(rgb_feat)
        t_s = self.sfs_t(t_feat)
        _, _, hs, ws = rgb_s.shape

        # 2) flatten + add pos embed
        rgb_tokens = self._to_tokens(rgb_s)
        t_tokens = self._to_tokens(t_s)
        pos = self._get_pos_embed(hs, ws)
        if pos is not None:
            rgb_tokens = rgb_tokens + pos
            t_tokens = t_tokens + pos

        # 3) iterative dual CFE enhancement
        rgb_hat_tokens, t_hat_tokens = self.icfe(rgb_tokens, t_tokens)

        # 4) reshape back to maps
        rgb_hat_s = self._to_map(rgb_hat_tokens, hs, ws)
        t_hat_s = self._to_map(t_hat_tokens, hs, ws)

        # 5) upsample to original resolution
        rgb_hat = F.interpolate(rgb_hat_s, size=(H, W), mode="bilinear", align_corners=False)
        t_hat = F.interpolate(t_hat_s, size=(H, W), mode="bilinear", align_corners=False)

        # 6) residual add with original features (as shown in the figure)
        rgb_hat = rgb_hat + rgb_feat
        t_hat = t_hat + t_feat

        # 7) final local fusion (NIN)
        fused = self.fusion(rgb_hat, t_hat)
        return fused # , rgb_hat, t_hat


if __name__ == "__main__":
    # Minimal usage example
    print("=== Testing DMFF Module ===")
    
    # 构造假数据: [Batch_size, Channels, Height, Width]
    rgb = torch.randn(2, 256, 80, 80)
    thermal = torch.randn(2, 256, 80, 80)
    
    print(f"输入 RGB 形状:     {rgb.shape}")
    print(f"输入 Thermal 形状: {thermal.shape}")
    
    # 初始化 DMFF 模块 (注意当前代码里要求 in_ch == out_ch)
    # 内部会将 80x80 缩小两倍计算注意力，再上采样还原
    dmff = DMFF(in_ch=256, out_ch=256)
    
    # 前向传播 (需要传入包含两个模态的列表)
    output = dmff([rgb, thermal])
    
    print(f"输出 融合特征 形状: {output.shape}")
    print("测试通过！可以正常输出 [B, C, H, W]")



