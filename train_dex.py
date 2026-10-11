#!/usr/bin/env python3
"""
Dex-TAG Training Pipeline
Trains the multi-finger contact mode and affordance-gated grasp network.
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models.dex_tag import DexTAG


class SyntheticDexDataset(Dataset):
    """
    Synthesizes diverse industrial objects with dual affordances:
    (1) Ergonomic safe grip handle
    (2) Functional working end
    Coupled with multi-finger contact mode ground truths:
    (0: Precision Pinch, 1: Tripod Dexterous, 2: Power Enveloping)
    """
    def __init__(self, num_samples=300, image_size=224):
        self.num_samples = num_samples
        self.image_size = image_size
        np.random.seed(42)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        H = W = self.image_size
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        depth = np.full((H, W), 0.70, dtype=np.float32)  # Table plane at 0.7m

        affordance_mask = np.zeros((H, W), dtype=np.float32)
        quality_geom = np.zeros((H, W), dtype=np.float32)
        sin_2theta = np.zeros((H, W), dtype=np.float32)
        cos_2theta = np.ones((H, W), dtype=np.float32)
        aperture = np.zeros((H, W), dtype=np.float32)
        contact_mode = np.zeros((H, W), dtype=np.int64)

        # Object center and orientation
        cx = np.random.randint(70, W - 70)
        cy = np.random.randint(70, H - 70)
        angle = np.random.uniform(0, math.pi)
        handle_len = np.random.randint(45, 65)
        head_len = np.random.randint(30, 45)

        dx = math.cos(angle)
        dy = math.sin(angle)
        nx = -dy
        ny = dx

        # Draw handle (Region 0: Safe Handle, Tripod/Pinch Mode)
        for t in range(-handle_len, 0):
            px = int(cx + t * dx)
            py = int(cy + t * dy)
            for w_off in range(-8, 9):
                qx = int(px + w_off * nx)
                qy = int(py + w_off * ny)
                if 0 <= qx < W and 0 <= qy < H:
                    rgb[qy, qx] = [0.15, 0.45, 0.9]  # Blue Handle
                    depth[qy, qx] = 0.65
                    affordance_mask[qy, qx] = 1.0     # Safe affordance
                    quality_geom[qy, qx] = 0.95
                    sin_2theta[qy, qx] = math.sin(2 * (angle + math.pi/2))
                    cos_2theta[qy, qx] = math.cos(2 * (angle + math.pi/2))
                    aperture[qy, qx] = 0.04           # 40mm grasp width
                    contact_mode[qy, qx] = 1          # Mode 1: Tripod Dexterous

        # Draw functional head (Region 1: Work Zone, Power/Pinch Mode)
        for t in range(0, head_len):
            px = int(cx + t * dx)
            py = int(cy + t * dy)
            for w_off in range(-14, 15):
                qx = int(px + w_off * nx)
                qy = int(py + w_off * ny)
                if 0 <= qx < W and 0 <= qy < H:
                    rgb[qy, qx] = [0.95, 0.35, 0.1]  # Orange Functional Head
                    depth[qy, qx] = 0.63
                    affordance_mask[qy, qx] = 0.0     # Hazardous for handover!
                    quality_geom[qy, qx] = 0.85       # Mechanically graspable
                    sin_2theta[qy, qx] = math.sin(2 * (angle + math.pi/2))
                    cos_2theta[qy, qx] = math.cos(2 * (angle + math.pi/2))
                    aperture[qy, qx] = 0.07
                    contact_mode[qy, qx] = 2          # Mode 2: Power Wrap

        # Add Gaussian noise
        rgb += np.random.normal(0, 0.02, rgb.shape).astype(np.float32)
        rgb = np.clip(rgb, 0.0, 1.0)
        depth += np.random.normal(0, 0.003, depth.shape).astype(np.float32)

        # Normalize RGB [-1, 1], Depth [0, 1]
        rgb_norm = (rgb - 0.5) / 0.5
        depth_norm = depth / 2.0
        rgbd = np.concatenate([rgb_norm, np.expand_dims(depth_norm, -1)], axis=-1)

        return {
            "x": torch.from_numpy(rgbd.transpose(2, 0, 1)).float(),
            "affordance_mask": torch.from_numpy(affordance_mask).unsqueeze(0).float(),
            "quality_geom": torch.from_numpy(quality_geom).unsqueeze(0).float(),
            "sin_2theta": torch.from_numpy(sin_2theta).unsqueeze(0).float(),
            "cos_2theta": torch.from_numpy(cos_2theta).unsqueeze(0).float(),
            "aperture": torch.from_numpy(aperture).unsqueeze(0).float(),
            "contact_mode": torch.from_numpy(contact_mode).long(),
        }


def train():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device='cuda') + 1
            device = torch.device('cuda')
        except Exception:
            device = torch.device('cpu')
    print(f"Training Dex-TAG on device: {device}")

    dataset = SyntheticDexDataset(num_samples=300)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = DexTAG(in_channels=4, base_channels=32, num_modes=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    num_epochs = 12
    best_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch['x'].to(device)
            target_aff = batch['affordance_mask'].to(device)
            target_q = batch['quality_geom'].to(device)
            target_sin = batch['sin_2theta'].to(device)
            target_cos = batch['cos_2theta'].to(device)
            target_ap = batch['aperture'].to(device)
            target_mode = batch['contact_mode'].to(device)

            out = model(x)

            l_aff = bce_loss(out['affordance_mask'], target_aff)
            l_q = mse_loss(out['quality_geom'], target_q)
            l_sin = mse_loss(out['sin_2theta'], target_sin)
            l_cos = mse_loss(out['cos_2theta'], target_cos)
            l_ap = mse_loss(out['aperture'], target_ap)
            l_mode = ce_loss(out['mode_logits'], target_mode)

            # AGAM consistency loss: quality_dex must align with affordance * quality
            target_dex = target_q * target_aff
            l_dex = mse_loss(out['quality_dex'], target_dex)

            loss = l_aff * 2.0 + l_q + l_sin + l_cos + l_ap + l_mode * 0.5 + l_dex * 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Loss: {avg_loss:.4f} (aff: {l_aff.item():.3f}, mode: {l_mode.item():.3f}, dex: {l_dex.item():.3f})")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join('checkpoints', 'dex_tag_best.pth')
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nTraining completed in {time.time() - start_time:.1f}s. Best weights saved to checkpoints/dex_tag_best.pth")


if __name__ == '__main__':
    train()
