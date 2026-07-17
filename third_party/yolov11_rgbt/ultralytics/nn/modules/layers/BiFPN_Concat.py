import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class SeparableConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.pointwise(self.depthwise(x)))


class BiFPN_Concat(nn.Module):
    def __init__(self, dimension=1, ch=[]):
        super().__init__()
        if not ch:
            raise ValueError("BiFPN_Concat requires 'ch' (input channels) to be initialized.")

        self.d = dimension
        self.input_channels = list(ch)
        self.n = len(self.input_channels)
        self.c2 = max(self.input_channels)
        self.epsilon = 1e-4
        self.w = nn.Parameter(torch.ones(self.n, dtype=torch.float32), requires_grad=True)
        self.resample = nn.ModuleList(
            Conv(c, self.c2, 1, 1, act=False) if c != self.c2 else nn.Identity() for c in self.input_channels
        )
        self.act = nn.SiLU()
        self.conv = SeparableConvBlock(self.c2)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = [x]

        if len(x) != self.n:
            raise ValueError(f"BiFPN_Concat expected {self.n} inputs, but received {len(x)}.")

        feats = [layer(feat) for layer, feat in zip(self.resample, x)]
        ref_shape = feats[0].shape[-2:]
        if any(feat.shape[-2:] != ref_shape for feat in feats[1:]):
            raise ValueError("BiFPN_Concat requires all inputs to have the same spatial shape before fusion.")

        weights = torch.relu(self.w)
        weights = weights / (weights.sum() + self.epsilon)
        weights = weights.to(device=feats[0].device, dtype=feats[0].dtype)

        out = weights[0] * feats[0]
        for weight, feat in zip(weights[1:], feats[1:]):
            out = out + weight * feat
        out = self.act(out)
        return self.conv(out)
