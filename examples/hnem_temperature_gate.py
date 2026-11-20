"""
HNEM temperature-gating component used in the manuscript experiments.

It covers neighborhood energy, a learnable positive temperature, two-modality
normalization, and fusion of paired real/imaginary high-frequency coefficients.
The downstream spatial-channel enhancement, IDTCWT, and detector are outside
this module.

Expected coefficient layout: [batch, channels, 6, height, width, 2].
The last axis contains the real and imaginary components, respectively.
"""

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class HNEMTemperatureGate(nn.Module):
    """Apply one shared learnable temperature to all stages that use this instance."""

    def __init__(self, tau_init: float, tau_floor: float = 1e-6) -> None:
        super().__init__()
        if tau_floor <= 0 or tau_init <= tau_floor:
            raise ValueError("Require 0 < tau_floor < tau_init")

        # Inverse softplus initializes softplus(theta) + tau_floor at tau_init.
        x = tau_init - tau_floor
        theta_init = x + math.log(-math.expm1(-x))
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        self.tau_floor = float(tau_floor)

    @staticmethod
    def neighborhood_energy(coefficients: torch.Tensor) -> torch.Tensor:
        """Sum |H|^2 over real/imaginary parts, 6 orientations, channels, and 3x3 space.

        Out-of-image locations contribute zero to boundary neighborhoods.
        Returns a tensor of shape [batch, 1, height, width].
        """
        magnitude_squared = coefficients.float().square().sum(dim=-1)
        pixel_energy = magnitude_squared.sum(dim=(1, 2)).unsqueeze(1)
        return F.avg_pool2d(
            pixel_energy,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=True,
        ) * 9.0

    def forward(
        self, h_ir: torch.Tensor, h_vis: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h_ir.shape != h_vis.shape or h_ir.ndim != 6:
            raise ValueError("h_ir and h_vis must have the same 6-D shape")
        if h_ir.shape[2] != 6 or h_ir.shape[-1] != 2:
            raise ValueError("Expected layout [B, C, 6, Hf, Wf, 2]")
        if not h_ir.is_floating_point() or not h_vis.is_floating_point():
            raise TypeError("High-frequency coefficients must be floating-point tensors")
        if h_ir.device != h_vis.device or h_ir.dtype != h_vis.dtype:
            raise ValueError("Both modalities must have the same device and dtype")

        e_ir = self.neighborhood_energy(h_ir)
        e_vis = self.neighborhood_energy(h_vis)

        # Equation (14): one scalar tau > 0, learned through the parameter theta.
        tau = F.softplus(self.theta) + self.tau_floor
        logits = torch.cat((e_ir, e_vis), dim=1) / tau  # [B, 2, Hf, Wf]
        m_ir = F.softmax(logits, dim=1)[:, 0:1]  # [B, 1, Hf, Wf]

        # Equation (15): broadcast the SAME real-valued weight over channels,
        # six orientations, and the paired real/imaginary components.
        weight = m_ir.to(h_ir.dtype).unsqueeze(2).unsqueeze(-1)
        h_fused = weight * h_ir + (1.0 - weight) * h_vis
        return h_fused, m_ir, tau


def _smoke_test() -> None:
    """Check algebra and gradients using synthetic coefficients."""
    # The experimental initialization is tau_init=20; these inputs are synthetic.
    gate = HNEMTemperatureGate(tau_init=20.0)
    h_ir = torch.zeros((1, 1, 6, 3, 3, 2), dtype=torch.float32)
    h_vis = torch.zeros_like(h_ir)
    h_ir[..., 0] = 1.0  # Infrared real components.
    h_vis[..., 1] = 0.5  # Visible imaginary components.

    h_fused, m_ir, tau = gate(h_ir, h_vis)
    center_weight = m_ir[0, 0, 1, 1]
    assert tau.item() > 0
    assert h_fused.shape == h_ir.shape
    assert torch.allclose(h_fused[0, 0, 0, 1, 1, 0], center_weight)
    assert torch.allclose(
        h_fused[0, 0, 0, 1, 1, 1], (1.0 - center_weight) * 0.5
    )

    h_fused.square().mean().backward()
    assert gate.theta.grad is not None and torch.isfinite(gate.theta.grad)
    optimizer = torch.optim.SGD(gate.parameters(), lr=0.1)
    optimizer.step()
    tau_after = F.softplus(gate.theta) + gate.tau_floor
    assert tau_after.item() > 0 and tau_after.item() != tau.item()
    print(f"Smoke-test initial tau: {tau.item():.6f}")
    print(f"Smoke-test tau after one optimizer step: {tau_after.item():.6f}")
    print(f"Smoke-test infrared weight at center: {center_weight.item():.6f}")
    print(f"Smoke-test gradient on learnable theta: {gate.theta.grad.item():.6e}")


if __name__ == "__main__":
    _smoke_test()
