import os
import math
import csv
import json

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from config import Config
from dataset import KolamPathDataset, collate_fn
from model import KolamModel
from loss import CompositeLoss
from eval_kolam19 import evaluate_held_out_40

# ──────────────────────────────────────────────────────────────────────────────
# LEARNING RATE & PRE-FLIGHT
# ──────────────────────────────────────────────────────────────────────────────

def get_lr_multiplier(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def pre_flight_verification(model, config):
    print("Pre-flight shape trace...")
    B, T = 2, config.max_seq_len
    dev  = config.device
    img  = torch.randn(B, 1, config.image_size, config.image_size).to(dev)
    tc   = torch.randn(B, T, 2).to(dev)
    pen  = torch.randint(0, 2, (B, T)).float().to(dev)
    pm   = torch.zeros((B, T), dtype=torch.bool).to(dev)
    with torch.no_grad():
        dp, el = model(img, tc, pen, pm)
        assert dp.shape == (B, T, 2), f"delta shape mismatch: {dp.shape}"
        assert el.shape == (B, T, 1), f"eos shape mismatch: {el.shape}"
    print("Pre-flight OK.")


def load_splits(splits_path="splits.json"):
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    print(f"Loaded {splits_path}: {splits['train_count']} train, {splits['held_count']} held-out ({splits.get('label')})")
    assert len(set(splits['train_images']).intersection(set(splits['held_images']))) == 0, "Overlap detected!"
    return splits


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def train():
    config = Config()
    os.makedirs("checkpoints", exist_ok=True)

    splits = load_splits("splits.json")

    train_dataset = KolamPathDataset(splits['train_images'], splits['train_csvs'], config)
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size,
                               collate_fn=collate_fn, shuffle=True, num_workers=0)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = config.num_epochs * steps_per_epoch

    print("Initializing fresh KolamModel (T=768, max_step=0.09, bounds=[-18.5, 18.5])...")
    model     = KolamModel(config).to(config.device)
    criterion = CompositeLoss(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.base_lr)

    pre_flight_verification(model, config)

    # Fixed start coordinate in normalized scale: (1.0, 0.0) -> norm: [0.054054, 0.0]
    c_range = config.coord_max - config.coord_min # 37.0
    start_norm_x = (1.0 - config.coord_min) / c_range * 2.0 - 1.0
    start_norm_y = (0.0 - config.coord_min) / c_range * 2.0 - 1.0
    print(f"Fixed logical start (1.0, 0.0) -> normalized: ({start_norm_x:.6f}, {start_norm_y:.6f})")

    # Gaussian noise scale: typical step is 1.118 lattice units = 1.118 / 18.5 = 0.0604 normalized
    # sigma = 0.3 * typical step = 0.3 * 0.0604 = ~0.0181
    noise_sigma = 0.3 * (1.118 / 18.5)
    print(f"Decoder coordinate input noise enabled: sigma = {noise_sigma:.6f}")

    log_path = "training_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'global_step', 'lr',
            'total_loss', 'coord_loss', 'eos_loss', 'curv_loss', 'closed_loop_loss',
            'raw_chamfer_mean', 'raw_chamfer_median',
            'raw_hausdorff_mean', 'raw_hausdorff_median',
            'raw_dtw_mean', 'raw_closure_mean', 'raw_beta1_mean',
            'snap_chamfer_mean', 'snap_chamfer_median',
            'snap_hausdorff_mean', 'snap_hausdorff_median',
            'snap_dtw_mean', 'snap_closure_mean', 'snap_beta1_mean',
            'gt_beta1'
        ])

    global_step = 0
    print(f"Starting training loop (Epoch 1 -> {config.num_epochs})...", flush=True)

    for epoch in range(config.num_epochs):
        model.train()
        epoch_totals = dict(total=0.0, coord=0.0, eos=0.0, curv=0.0, loop=0.0)
        n_batches = 0

        for batch in train_loader:
            image         = batch['image'].to(config.device)
            target_coords = batch['target_coords'].to(config.device)
            pen_state     = batch['pen_state'].to(config.device)
            pad_mask      = batch['pad_mask'].to(config.device)
            seq_len       = batch['seq_len'].to(config.device)

            optimizer.zero_grad()

            # Teacher forcing = 1.0 throughout
            # Add Gaussian noise (sigma = 0.3 * typical step) to decoder input coordinates
            noise = torch.randn_like(target_coords) * noise_sigma
            # Do not corrupt padded tokens
            noise = noise * (~pad_mask.unsqueeze(-1)).float()
            # Preserve exact starting coordinate at t=0
            noise[:, 0, :] = 0.0
            noisy_decoder_input = target_coords + noise

            delta_preds, eos_logits = model(image, noisy_decoder_input, pen_state, pad_mask)
            shared_param = model.decoder.transformer_decoder.layers[-1].linear2.weight
            loss, loss_dict = criterion(
                delta_preds, eos_logits, target_coords, pen_state, pad_mask, seq_len,
                global_step=global_step, shared_trunk_param=shared_param
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            # Schedule update
            lr_mult = get_lr_multiplier(global_step, config.warmup_steps, total_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = config.base_lr * lr_mult

            epoch_totals['total'] += loss.item()
            epoch_totals['coord'] += loss_dict['coord_loss']
            epoch_totals['eos']   += loss_dict['eos_loss']
            epoch_totals['curv']  += loss_dict['curv_loss']
            epoch_totals['loop']  += loss_dict['closed_loop_loss']
            n_batches += 1
            global_step += 1

        nb = max(1, n_batches)
        avg = {k: v / nb for k, v in epoch_totals.items()}
        current_lr = optimizer.param_groups[0]['lr']

        # ── Eval on 40 held-out images every epoch ─────────────────────────────
        print(f"Evaluating 40 held-out images for Epoch {epoch+1}...", flush=True)
        eval_metrics = evaluate_held_out_40(model, config, splits_path="splits.json")
        raw_m = eval_metrics['raw']
        snp_m = eval_metrics['snapped']
        gt_b1 = eval_metrics['gt_beta1']

        print(
            f"Epoch {epoch+1:3d}/{config.num_epochs} | "
            f"Loss {avg['total']:.4f} | Coord {avg['coord']:.5f} | EOS {avg['eos']:.4f} | "
            f"Raw Chamf {raw_m['chamfer_mean']:.3f} | Snap Chamf {snp_m['chamfer_mean']:.3f} | "
            f"Raw Clos {raw_m['closure_mean']:.3f} | Snap Clos {snp_m['closure_mean']:.3f} | "
            f"LR {current_lr:.2e}",
            flush=True
        )

        # ── Checkpoint every 5 epochs ──────────────────────────────────────────
        if (epoch + 1) % 5 == 0 or (epoch + 1) == 1:
            ckpt_path = f"checkpoints/checkpoint_epoch_{epoch+1:03d}.pt"
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'eval_metrics': eval_metrics
            }, ckpt_path)
            print(f"  >> Saved {ckpt_path}", flush=True)

        # ── Log row ────────────────────────────────────────────────────────────
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, global_step, current_lr,
                avg['total'], avg['coord'], avg['eos'], avg['curv'], avg['loop'],
                raw_m['chamfer_mean'], raw_m['chamfer_median'],
                raw_m['hausdorff_mean'], raw_m['hausdorff_median'],
                raw_m['dtw_mean'], raw_m['closure_mean'], raw_m['beta1_mean'],
                snp_m['chamfer_mean'], snp_m['chamfer_median'],
                snp_m['hausdorff_mean'], snp_m['hausdorff_median'],
                snp_m['dtw_mean'], snp_m['closure_mean'], snp_m['beta1_mean'],
                gt_b1
            ])

    print("Scoped training run complete.", flush=True)


if __name__ == '__main__':
    train()
