"""
Dex-TAG: Dexterous Task-Aware Grasp Network
Paper Contribution:
1. Multi-Finger Contact Mode Taxonomy (Precision Pinch, Tripod Dexterous, Power Enveloping)
2. Affordance-Gated Attention Module (AGAM) for Intent-Conditioned Grasp Synthesis
3. Dual-Stream Spatial Feature Fusion for Contact Mechanics and Semantic Clearance
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class AffordanceGatedAttentionModule(nn.Module):
    """
    AGAM: Differentiable bilateral attention gate conditioning geometric contact
    mechanics on semantic functional affordances.
    """
    def __init__(self, channels):
        super().__init__()
        self.gate_projector = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, f_geom, f_aff, q_geom):
        f_cat = torch.cat([f_geom, f_aff], dim=1)
        soft_gate = self.gate_projector(f_cat)
        q_dex = q_geom * soft_gate
        return q_dex, soft_gate


class DexTAG(nn.Module):
    """
    Dexterous Task-Aware Grasp Network.
    Outputs:
        - affordance_mask: Semantic functional zone [B, 1, H, W]
        - quality_geom: Antipodal force-closure quality [B, 1, H, W]
        - quality_dex: Task-conditioned affordance-gated quality [B, 1, H, W]
        - sin_2theta: Grasp orientation sine component [B, 1, H, W]
        - cos_2theta: Grasp orientation cosine component [B, 1, H, W]
        - aperture: Contact aperture / finger spread [B, 1, H, W]
        - contact_modes: Multi-finger mode logits (Pinch, Tripod, Power) [B, 3, H, W]
        - soft_gate: Attention modulation map [B, 1, H, W]
    """
    CONTACT_MODES = ['PRECISION_PINCH', 'TRIPOD_DEXTEROUS', 'POWER_ENVELOPING']

    def __init__(self, in_channels=4, base_channels=32, num_modes=3):
        super().__init__()
        self.num_modes = num_modes

        # 1. Multi-Scale Shared Feature Encoder
        self.stem = ConvBNReLU(in_channels, base_channels, kernel_size=3, stride=1)
        self.stage1 = nn.Sequential(
            InvertedResidual(base_channels, base_channels, stride=1, expand_ratio=1),
            InvertedResidual(base_channels, base_channels * 2, stride=2, expand_ratio=3)
        )
        self.stage2 = nn.Sequential(
            InvertedResidual(base_channels * 2, base_channels * 2, stride=1, expand_ratio=3),
            InvertedResidual(base_channels * 2, base_channels * 4, stride=2, expand_ratio=4)
        )
        self.bottleneck = nn.Sequential(
            InvertedResidual(base_channels * 4, base_channels * 4, stride=1, expand_ratio=4),
            InvertedResidual(base_channels * 4, base_channels * 4, stride=1, expand_ratio=4)
        )

        # 2. Decoder Stream A: Semantic Affordance Field
        self.aff_up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.aff_up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.aff_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

        # 3. Decoder Stream B: Dexterous Contact Mechanics Field
        self.geom_up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.geom_up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Geometric & Contact Heads
        self.geom_quality_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.sin_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.cos_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.aperture_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.mode_head = nn.Conv2d(base_channels, num_modes, kernel_size=3, padding=1)

        # 4. Affordance-Gated Attention Module
        self.agam = AffordanceGatedAttentionModule(base_channels)

    def forward(self, x):
        # Shared Representation
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        bottleneck = self.bottleneck(feat)

        # Semantic Affordance Stream
        f_aff = self.aff_up1(bottleneck)
        f_aff = self.aff_up2(f_aff)
        affordance_mask = torch.sigmoid(self.aff_head(f_aff))

        # Dexterous Contact Stream
        f_geom = self.geom_up1(bottleneck)
        f_geom = self.geom_up2(f_geom)
        quality_geom = torch.sigmoid(self.geom_quality_head(f_geom))
        sin_2theta = torch.tanh(self.sin_head(f_geom))
        cos_2theta = torch.tanh(self.cos_head(f_geom))
        aperture = torch.sigmoid(self.aperture_head(f_geom))
        mode_logits = self.mode_head(f_geom)
        contact_modes = F.softmax(mode_logits, dim=1)

        # Affordance-Gated Attention Modulation
        quality_dex, soft_gate = self.agam(f_geom, f_aff, quality_geom)

        return {
            "affordance_mask": affordance_mask,
            "quality_geom": quality_geom,
            "quality_dex": quality_dex,
            "sin_2theta": sin_2theta,
            "cos_2theta": cos_2theta,
            "aperture": aperture,
            "mode_logits": mode_logits,
            "contact_modes": contact_modes,
            "soft_gate": soft_gate
        }
