from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
except ImportError:
    DTCWTForward = DTCWTInverse = None


def _FZJS(channels: int, preferred: int = 8) -> int:
    for groups in range(min(channels, preferred), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _JGHJH(nn.Sequential):

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, act: bool = True):
        p = k // 2
        layers: list[nn.Module] = [nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False), nn.BatchNorm2d(c2)]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class _CKJHCYL(nn.Module):

    def __init__(self, channels: int, reduction: int = 4, pool: int = 1):
        super().__init__()
        if reduction < 1 or pool < 1:
            raise ValueError("reduction and pool must be positive integers")
        self.hidden = max(channels // reduction, 8)
        self.pool = int(pool)
        self.query = nn.Conv2d(channels, self.hidden, 1, bias=False)
        self.key = nn.Conv2d(channels, self.hidden, 1, bias=False)
        self.value = nn.Conv2d(channels, self.hidden, 1, bias=False)
        self.output = nn.Conv2d(self.hidden, channels, 1, bias=False)
        self.scale = self.hidden ** -0.5

    def _reduce_tokens(self, x: Tensor) -> Tensor:
        if self.pool == 1:
            return x
        return F.avg_pool2d(x, kernel_size=self.pool, stride=self.pool, ceil_mode=True)

    def forward(self, target: Tensor, auxiliary: Tensor) -> Tensor:
        if target.shape != auxiliary.shape:
            raise ValueError(f"SJCZYLKJMKRH expects paired equal-shaped features, got {target.shape} and {auxiliary.shape}")
        h, w = target.shape[-2:]
        target_small = self._reduce_tokens(target)
        auxiliary_small = self._reduce_tokens(auxiliary)
        b, _, hs, ws = target_small.shape
        n = hs * ws

        q = self.query(target_small).flatten(2).transpose(1, 2)
        k = self.key(auxiliary_small).flatten(2)
        v = self.value(auxiliary_small).flatten(2).transpose(1, 2)
        attention = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        output = torch.matmul(attention, v).transpose(1, 2).reshape(b, self.hidden, hs, ws)
        output = self.output(output)
        return F.interpolate(output, size=(h, w), mode="bilinear", align_corners=False) if (hs, ws) != (h, w) else output


class SJCZYLKJMKRH(nn.Module):

    def __init__(self, channels: int, reduction: int = 4, attn_pool: int = 4):
        super().__init__()
        self.vis_from_ir = _CKJHCYL(channels, reduction, attn_pool)
        self.ir_from_vis = _CKJHCYL(channels, reduction, attn_pool)
        self.lambda_vis = nn.Parameter(torch.ones(1))
        self.mu_vis = nn.Parameter(torch.ones(1))
        self.lambda_ir = nn.Parameter(torch.ones(1))
        self.mu_ir = nn.Parameter(torch.ones(1))
        self.norm_vis = nn.LayerNorm(channels, eps=1e-5)
        self.norm_ir = nn.LayerNorm(channels, eps=1e-5)
        self.spatial_gate = nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=True)

    @staticmethod
    def _token_norm(x: Tensor, norm: nn.LayerNorm) -> Tensor:
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: Sequence[Tensor]) -> Tensor:
        if len(x) != 2:
            raise ValueError("SJCZYLKJMKRH expects [visible_feature, infrared_feature]")
        visible, infrared = x
        visible_hat = self._token_norm(
            self.lambda_vis * visible + self.mu_vis * self.vis_from_ir(visible, infrared), self.norm_vis
        )
        infrared_hat = self._token_norm(
            self.lambda_ir * infrared + self.mu_ir * self.ir_from_vis(infrared, visible), self.norm_ir
        )
        joint = torch.cat((visible_hat, infrared_hat), dim=1)
        gate_input = torch.cat((joint.mean(1, keepdim=True), joint.amax(1, keepdim=True)), dim=1)
        modality_stats = torch.cat(
            (
                visible_hat.mean(1, keepdim=True) - infrared_hat.mean(1, keepdim=True),
                visible_hat.amax(1, keepdim=True) - infrared_hat.amax(1, keepdim=True),
            ),
            dim=1,
        )
        gate = torch.sigmoid(self.spatial_gate(torch.cat((gate_input, modality_stats), dim=1)))
        return gate * visible_hat + (1.0 - gate) * infrared_hat


class _FXCBCH(nn.Module):

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        norm_groups = _FZJS(channels, groups)
        self.target_map = nn.Conv2d(channels, channels, 1, bias=False)
        self.inject_map = nn.Conv2d(channels, channels, 1, bias=False)
        self.height_gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels, bias=False),
            nn.GroupNorm(norm_groups, channels),
            nn.Sigmoid(),
        )
        self.width_gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels, bias=False),
            nn.GroupNorm(norm_groups, channels),
            nn.Sigmoid(),
        )

    def forward(self, target: Tensor, auxiliary: Tensor) -> Tensor:
        residual = auxiliary - self.target_map(target)
        h_stats = torch.cat((residual.mean(dim=3, keepdim=True), residual.var(dim=3, keepdim=True, unbiased=False).sqrt()), 1)
        w_stats = torch.cat((residual.mean(dim=2, keepdim=True), residual.var(dim=2, keepdim=True, unbiased=False).sqrt()), 1)
        structural_map = self.height_gate(h_stats) * self.width_gate(w_stats)
        return target + self.inject_map(auxiliary * structural_map)


class GPLYNLMTXZ(nn.Module):

    def __init__(self, tau_init: float = 20.0, tau_floor: float = 1e-6):
        super().__init__()
        if tau_init <= tau_floor or tau_floor <= 0:
            raise ValueError("require 0 < tau_floor < tau_init")
        x = tau_init - tau_floor
        self.theta = nn.Parameter(torch.tensor(x + math.log(-math.expm1(-x)), dtype=torch.float32))
        self.tau_floor = float(tau_floor)

    @staticmethod
    def neighborhood_energy(coefficients: Tensor) -> Tensor:
        magnitude_squared = coefficients.float().square().sum(dim=-1)
        per_pixel = magnitude_squared.sum(dim=(1, 2)).unsqueeze(1)
        return F.avg_pool2d(per_pixel, 3, stride=1, padding=1, count_include_pad=True) * 9.0

    def forward(self, h_ir: Tensor, h_vis: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if h_ir.shape != h_vis.shape or h_ir.ndim != 6 or h_ir.shape[2] != 6 or h_ir.shape[-1] != 2:
            raise ValueError("HNEM expects equal DTCWT tensors in [B, C, 6, H, W, 2] layout")
        e_ir, e_vis = self.neighborhood_energy(h_ir), self.neighborhood_energy(h_vis)
        tau = F.softplus(self.theta) + self.tau_floor
        m_ir = torch.softmax(torch.cat((e_ir, e_vis), dim=1) / tau, dim=1)[:, :1]
        weight = m_ir.to(dtype=h_ir.dtype).unsqueeze(2).unsqueeze(-1)
        return weight * h_ir + (1.0 - weight) * h_vis, m_ir, tau


class _KJTQZQ(nn.Module):

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.spatial_3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.spatial_5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.spatial_project = nn.Conv2d(2 * channels, channels, 1, bias=True)
        self.channel_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        spatial = torch.sigmoid(self.spatial_project(torch.cat((self.spatial_3(x), self.spatial_5(x)), 1)))
        channel = torch.sigmoid(self.channel_gate(x))
        return x * spatial * channel


class PLGZSFZHBZQK(nn.Module):

    def __init__(self, channels: int, groups: int = 8, tau_init: float = 20.0):
        super().__init__()
        if DTCWTForward is None or DTCWTInverse is None:
            raise ImportError("PLGZSFZHBZQK requires pytorch_wavelets==1.3.0. Install it before creating CHSF-Net.")
        self.forward_dtcwt = DTCWTForward(
            J=1, biort="near_sym_a", qshift="qshift_a", include_scale=False, o_dim=2, ri_dim=-1
        )
        self.inverse_dtcwt = DTCWTInverse(biort="near_sym_a", qshift="qshift_a", o_dim=2, ri_dim=-1)
        self.ir_low = _FXCBCH(channels, groups)
        self.vis_low = _FXCBCH(channels, groups)
        self.low_fuse = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.hnem = GPLYNLMTXZ(tau_init=tau_init)
        self.high_enhance = _KJTQZQ(channels)

    def forward(self, x: Sequence[Tensor]) -> Tensor:
        if len(x) != 2:
            raise ValueError("PLGZSFZHBZQK expects [visible_feature, infrared_feature]")
        visible, infrared = x
        if visible.shape != infrared.shape:
            raise ValueError(f"PLGZSFZHBZQK expects paired equal-shaped features, got {visible.shape} and {infrared.shape}")
        height, width = visible.shape[-2:]
        low_vis, high_vis = self.forward_dtcwt(visible)
        low_ir, high_ir = self.forward_dtcwt(infrared)
        if len(high_vis) != 1 or len(high_ir) != 1:
            raise RuntimeError("PLGZSFZHBZQK is defined for a single DTCWT decomposition level")
        low_ir_hat = self.ir_low(low_ir, low_vis)
        low_vis_hat = self.vis_low(low_vis, low_ir)
        high_fused, _, _ = self.hnem(high_ir[0], high_vis[0])
        high_hat = self.high_enhance(high_fused)
        low_fused = self.low_fuse(torch.cat((low_ir_hat, low_vis_hat), dim=1))
        reconstructed = self.inverse_dtcwt((low_fused, [high_hat]))
        return reconstructed[..., :height, :width]


class DTT(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.alpha * x) + self.beta


class _TZDQQ(nn.Module):

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.project = _JGHJH(c1, c2, 1)

    def forward(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        x = self.project(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False) if x.shape[-2:] != size else x


class _DCCYY(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.project = _JGHJH(2 * channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.project(torch.cat((self.dw3(x), self.dw5(x)), dim=1))


class DTKJPLHH(nn.Module):

    def __init__(self, c_freq: int, c_skip: int, c_shallow: int, c_out: int, groups: int = 8):
        super().__init__()
        self.freq_align = _TZDQQ(c_freq, c_out)
        self.skip_align = _TZDQQ(c_skip, c_out)
        self.shallow_align = _TZDQQ(c_shallow, c_out)
        self.mix1 = _JGHJH(3 * c_out, c_out, 1)
        self.dyt1 = DTT(c_out)
        self.mix2 = _JGHJH(c_out, c_out, 1)
        self.dyt2 = DTT(c_out)
        self.att_freq = nn.Conv2d(c_out, c_out, 1, bias=True)
        self.att_skip = nn.Conv2d(c_out, c_out, 1, bias=True)
        self.context = _DCCYY(c_out)
        self.shortcut = _JGHJH(3 * c_out, c_out, 1, act=False)
        self.output_dyt = DTT(c_out)

    def forward(self, x: Sequence[Tensor]) -> Tensor:
        if len(x) != 3:
            raise ValueError("DTKJPLHH expects [frequency_enhanced, top_down_semantic, shallow_cross_modal]")
        frequency, semantic, shallow = x
        target_size = frequency.shape[-2:]
        frequency = self.freq_align(frequency, target_size)
        semantic = self.skip_align(semantic, target_size)
        shallow = self.shallow_align(shallow, target_size)
        concatenated = torch.cat((frequency, semantic, shallow), dim=1)
        fused = F.relu(self.dyt1(self.mix1(concatenated)), inplace=False)
        fused = F.relu(self.dyt2(self.mix2(fused)), inplace=False)
        spatial_attention = torch.sigmoid(self.att_freq(frequency) + self.att_skip(semantic))
        enhanced = self.context(spatial_attention * fused)
        return self.output_dyt(enhanced + self.shortcut(concatenated))
