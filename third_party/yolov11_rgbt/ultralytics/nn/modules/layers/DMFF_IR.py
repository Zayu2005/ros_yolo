import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedPool2d(nn.Module):
    """
    SFS pooling:
        F_a = AvgPool(F)
        F_m = MaxPool(F)
        F_o = lambda * F_a + (1-lambda) * F_m
    where lambda is learnable.
    """

    def __init__(self, kernel_size: int = 2, stride: Optional[int] = None, init_lambda: float = 0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
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
    Uses PixelUnshuffle to move spatial info into channel dimension,
    then compresses channels with 1x1 conv.
    """

    def __init__(self, channels: int, scale: int = 2):
        super().__init__()
        assert scale >= 1 and isinstance(scale, int)
        self.scale = scale
        if scale == 1:
            self.op = nn.Identity()
        else:
            self.op = nn.Sequential(
                nn.PixelUnshuffle(scale),  # B, C*s^2, H/s, W/s
                nn.Conv2d(channels * scale * scale, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SFS(nn.Module):
    """
    Spatial Feature Shrinking module.

    mode='pool'  -> mixed avg/max pooling
    mode='conv'  -> spatial-to-channel + 1x1 conv
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
    Cross-attention enhancement:
      - Q from aux_tokens
      - K,V from tgt_tokens
      - output stays in tgt_tokens space

    In this modified version, we only use this to enhance the IR branch
    with RGB as auxiliary guidance.
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

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, C] -> [B, h, N, d]
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, h, N, d] -> [B, N, C]
        B, h, N, d = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, N, h * d)

    def forward(self, aux_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        aux_tokens: [B, N, C]  e.g. RGB tokens
        tgt_tokens: [B, N, C]  e.g. IR tokens
        returns:    [B, N, C]  enhanced target tokens
        """
        q = self._reshape_heads(self.q_proj(aux_tokens))
        k = self._reshape_heads(self.k_proj(tgt_tokens))
        v = self._reshape_heads(self.v_proj(tgt_tokens))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        z = torch.matmul(attn, v)
        z = self._merge_heads(z)
        z = self.out_proj(z)

        t_prime = self.alpha * z + self.beta * tgt_tokens
        out = self.gamma * t_prime + self.delta * self.ffn(t_prime)
        return out


class IRGuidedICFE(nn.Module):
    """
    Modified ICFE:
    Only RGB -> IR enhancement.
    No reverse IR -> RGB enhancement, to avoid RGB noise polluting the stronger IR branch.
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
        self.cfe_ir = CrossFeatureEnhancement(dim, num_heads, mlp_ratio, dropout)

    def forward(self, rgb_tokens: torch.Tensor, ir_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = rgb_tokens
        t = ir_tokens
        for _ in range(self.num_iters):
            # keep RGB tokens unchanged, only update IR tokens
            t = self.cfe_ir(aux_tokens=r, tgt_tokens=t)
        return r, t


class GatedIRFusion(nn.Module):
    """
    IR-dominant gated fusion:
        fused = ir + gate * rgb
    gate is generated from [ir, rgb].
    """

    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(torch.cat([ir_feat, rgb_feat], dim=1))
        fused = ir_feat + gate * rgb_feat
        fused = self.proj(fused)
        return fused, gate


class DMFF_IR(nn.Module):
    """
    Modified DMFF for your task: IR-dominant dual-modal fusion.

    Main changes vs original version:
      1) only RGB -> IR cross enhancement (single-direction)
      2) RGB branch is preserved, not updated by IR
      3) final fusion uses gated IR-dominant residual fusion instead of symmetric concat fusion

    Pipeline:
      1) SFS on RGB and IR feature maps
      2) flatten + positional embedding -> tokens
      3) IR-guided ICFE: RGB guides IR only
      4) reshape back to feature maps
      5) bilinear upsample to original spatial size
      6) residual add with original modality features
      7) gated IR-dominant fusion:
             fused = ir_hat + gate * rgb_hat
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
        assert in_ch == out_ch, f"DMFF_IR中in_ch:{in_ch} != out_ch:{out_ch}"
        channels = in_ch
        self.channels = channels
        self.use_pos_embed = use_pos_embed
        self.max_tokens_hw = max_tokens_hw

        self.sfs_rgb = SFS(channels, scale=sfs_scale, mode=sfs_mode)
        self.sfs_ir = SFS(channels, scale=sfs_scale, mode=sfs_mode)

        self.icfe = IRGuidedICFE(
            dim=channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_iters=num_iters,
        )

        self.fusion = GatedIRFusion(channels, out_channels=out_channels or channels)

        if use_pos_embed:
            if max_tokens_hw is None:
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
        rgb_feat, ir_feat = x[0], x[1]
        assert rgb_feat.shape == ir_feat.shape, "RGB/IR feature shapes must match"
        B, C, H, W = rgb_feat.shape
        assert C == self.channels, f"Expected channels={self.channels}, got {C}"

        # 1) spatial shrinking
        rgb_s = self.sfs_rgb(rgb_feat)
        ir_s = self.sfs_ir(ir_feat)
        _, _, hs, ws = rgb_s.shape

        # 2) flatten + positional encoding
        rgb_tokens = self._to_tokens(rgb_s)
        ir_tokens = self._to_tokens(ir_s)

        pos = self._get_pos_embed(hs, ws)
        if pos is not None:
            rgb_tokens = rgb_tokens + pos
            ir_tokens = ir_tokens + pos

        # 3) only RGB -> IR enhancement
        rgb_hat_tokens, ir_hat_tokens = self.icfe(rgb_tokens, ir_tokens)

        # 4) reshape back
        rgb_hat_s = self._to_map(rgb_hat_tokens, hs, ws)
        ir_hat_s = self._to_map(ir_hat_tokens, hs, ws)

        # 5) upsample to original spatial size
        rgb_hat = F.interpolate(rgb_hat_s, size=(H, W), mode="bilinear", align_corners=False)
        ir_hat = F.interpolate(ir_hat_s, size=(H, W), mode="bilinear", align_corners=False)

        # 6) residual add
        rgb_hat = rgb_hat + rgb_feat
        ir_hat = ir_hat + ir_feat

        # 7) IR-dominant gated fusion
        fused, gate = self.fusion(rgb_hat, ir_hat)

        # return gate for visualization/debugging if you want to inspect RGB contribution
        return fused # , rgb_hat, ir_hat, gate


if __name__ == "__main__":
    rgb = torch.randn(2, 256, 80, 80)
    ir = torch.randn(2, 256, 80, 80)

    dmff = DMFF_IR(
        in_ch=256,
        out_ch=256,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        sfs_scale=2,
        sfs_mode="pool",
        num_iters=1,
        max_tokens_hw=(40, 40),
        use_pos_embed=True,
    )

    # fused, rgb_enh, ir_enh, gate = dmff(rgb, ir)
    fused = dmff([rgb, ir])
    print("rgb:", rgb.shape)
    print("ir:", ir.shape)
    # print("rgb_enh:", rgb_enh.shape)
    # print("ir_enh:", ir_enh.shape)
    # print("gate:", gate.shape)
    print("fused:", fused.shape)