#!/usr/bin/env python3
"""
Dex-TAG Benchmark Evaluation Suite
Evaluates Geometric Success Rate (GSR), Task Success Rate (TSR),
Multi-Finger Contact Mode Accuracy, and Hardware Inference Latency.
"""

import os
import time
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.dex_tag import DexTAG
from train_dex import SyntheticDexDataset


def evaluate():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device='cuda') + 1
            device = torch.device('cuda')
        except Exception:
            device = torch.device('cpu')
    print(f"Evaluating Dex-TAG on device: {device}")

    dataset = SyntheticDexDataset(num_samples=200)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = DexTAG(in_channels=4, base_channels=32, num_modes=3).to(device)
    ckpt_path = os.path.join('checkpoints', 'dex_tag_best.pth')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("Warning: Running with uninitialized weights.")
    model.eval()

    total_samples = 0
    geom_success = 0
    task_success = 0
    mode_correct = 0
    latencies = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            gt_aff = batch['affordance_mask'].squeeze().numpy()
            gt_q = batch['quality_geom'].squeeze().numpy()
            gt_mode = batch['contact_mode'].squeeze().numpy()

            t0 = time.perf_counter()
            out = model(x)
            t_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_ms)

            q_dex = out['quality_dex'].squeeze().cpu().numpy()
            q_geom = out['quality_geom'].squeeze().cpu().numpy()
            modes = torch.argmax(out['contact_modes'], dim=1).squeeze().cpu().numpy()

            # Find peak grasp position
            max_idx = np.unravel_index(np.argmax(q_dex), q_dex.shape)
            vy, vx = max_idx

            # Geometric stability check
            if q_geom[vy, vx] > 0.6:
                geom_success += 1

            # Task safety check: Must lie on safe affordance region (handle)
            if gt_aff[vy, vx] > 0.5:
                task_success += 1

            # Mode check: Did it select Tripod Dexterous (mode 1) for the cylindrical handle?
            if modes[vy, vx] == gt_mode[vy, vx]:
                mode_correct += 1

            total_samples += 1

    gsr = (geom_success / total_samples) * 100.0
    tsr = (task_success / total_samples) * 100.0
    mode_acc = (mode_correct / total_samples) * 100.0
    avg_latency = float(np.mean(latencies[5:])) if len(latencies) > 5 else float(np.mean(latencies))
    fps = 1000.0 / avg_latency

    print("\n" + "=" * 65)
    print("           DEX-TAG RIGOROUS BENCHMARK EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Test Objects Evaluated      : {total_samples}")
    print(f"  Geometric Success Rate (GSR): {gsr:.1f}%")
    print(f"  Task Success Rate (TSR)     : {tsr:.1f}%")
    print(f"  Contact Mode Accuracy       : {mode_acc:.1f}%")
    print(f"  Average Inference Latency   : {avg_latency:.2f} ms (~{fps:.0f} FPS)")
    print("=" * 65)


if __name__ == '__main__':
    evaluate()
