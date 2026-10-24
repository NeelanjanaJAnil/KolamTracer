import os
import json
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import Config
from model import KolamModel
from dataset import KolamPathDataset, denormalize
from topology import render_path_to_binary, compute_betti_numbers
from spectral import choose_K, fit_fourier, eval_fourier, resample_uniform_arclength

def compute_hausdorff(pts1, pts2):
    from scipy.spatial.distance import cdist
    dists = cdist(pts1, pts2)
    d_12 = np.max(np.min(dists, axis=1))
    d_21 = np.max(np.min(dists, axis=0))
    return float(max(d_12, d_21))

def compute_rms(pts1, pts2):
    p1 = resample_uniform_arclength(pts1, n=512)
    p2 = resample_uniform_arclength(pts2, n=512)
    return float(np.sqrt(np.mean(np.sum((p1 - p2) ** 2, axis=1))))

def run_validation(ckpt_path=None, splits_path="splits.json"):
    config = Config()
    
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Missing {splits_path}! train.py must create it first.")
        
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        
    held_images = splits['held_images']
    held_csvs = splits['held_csvs']
    held_hash = splits.get('held_out_hash')
    N = len(held_images)
    print(f"Loaded {N} held-out samples from {splits_path} (Hash: {held_hash})")
    
    val_dataset = KolamPathDataset(held_images, held_csvs, config)
    
    # Checkpoint selection: default to latest in checkpoints/
    if ckpt_path is None:
        import glob
        ckpts = sorted(glob.glob("checkpoints/checkpoint_epoch_*.pt"))
        if ckpts:
            ckpt_path = ckpts[-1]
        else:
            ckpt_path = "checkpoint_epoch_015.pt"
            
    model = KolamModel(config).to(config.device)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
        # Assert split hash matches checkpoint if stored
        if 'held_out_hash' in checkpoint and checkpoint['held_out_hash'] is not None:
            assert checkpoint['held_out_hash'] == held_hash, f"Hash mismatch! Checkpoint: {checkpoint['held_out_hash']}, Splits: {held_hash}"
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"Warning: {ckpt_path} not found. Running with uninitialized model.")
    model.eval()
    
    results = []
    
    print(f"\n--- Running Validation Across {N} Stratified Held-Out Samples ---")
    for i in range(N):
        sample = val_dataset[i]
        csv_file = os.path.basename(held_csvs[i])
        
        gt_norm = sample['target_coords'].numpy()
        seq_len = int(sample['seq_len'].item()) + 1
        gt_norm_valid = gt_norm[:seq_len]
        gt_pts = denormalize(gt_norm_valid, config=config)
        
        # Greedy rollout with true start coordinate
        with torch.no_grad():
            img_tensor = sample['image'].unsqueeze(0).to(config.device)
            coords_tensor = sample['target_coords'].unsqueeze(0).to(config.device)
            start_coords = coords_tensor[:, 0:1, :]
            memory = model.encoder(img_tensor)
            pred_coords, _ = model.decoder.autoregressive_generate(
                memory, coords_tensor.shape[1], start_coords=start_coords
            )
        pred_norm = pred_coords[0].cpu().numpy()[:seq_len]
        model_pts = denormalize(pred_norm, config=config)
        
        chosen_K, is_match, metrics = choose_K(model_pts, gt_pts, eps_lattice=0.25, k_max=80)
        
        f_coeffs = fit_fourier(model_pts, chosen_K)
        t_eval = np.linspace(0, 2*np.pi, 512, endpoint=False)
        fx, fy = eval_fourier(f_coeffs, t_eval)
        fourier_pts = np.column_stack([fx, fy])
        
        rms_vs_gt = compute_rms(fourier_pts, gt_pts)
        hausdorff_vs_gt = compute_hausdorff(fourier_pts, gt_pts)
        
        rms_vs_model = compute_rms(fourier_pts, model_pts)
        hausdorff_vs_model = compute_hausdorff(fourier_pts, model_pts)
        
        gt_bin = render_path_to_binary(gt_pts[:, 0], gt_pts[:, 1], img_size=128)
        gt_b0, gt_b1 = compute_betti_numbers(gt_bin)
        
        f_bin = render_path_to_binary(fourier_pts[:, 0], fourier_pts[:, 1], img_size=128)
        fb0, fb1 = compute_betti_numbers(f_bin)
        
        b0_exact_match = (fb0 == gt_b0)
        b1_exact_match = (fb1 == gt_b1)
        
        results.append({
            'sample_idx': i + 1,
            'csv_name': csv_file,
            'chosen_K': chosen_K,
            'rms_vs_gt': rms_vs_gt,
            'hausdorff_vs_gt': hausdorff_vs_gt,
            'rms_vs_model': rms_vs_model,
            'hausdorff_vs_model': hausdorff_vs_model,
            'gt_beta0': gt_b0,
            'gt_beta1': gt_b1,
            'fourier_beta0': fb0,
            'fourier_beta1': fb1,
            'beta0_match': b0_exact_match,
            'beta1_match': b1_exact_match,
            'exact_topo_match': (b0_exact_match and b1_exact_match)
        })
        
    df_res = pd.DataFrame(results)
    os.makedirs("reports", exist_ok=True)
    csv_out = os.path.join("reports", "validation.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\nSaved validation results to {csv_out}")
    
    print("\n================== VALIDATION SUMMARY (N=30) ==================")
    print(f"RMS vs GT (lattice units):     Mean = {df_res['rms_vs_gt'].mean():.3f}, Median = {df_res['rms_vs_gt'].median():.3f}")
    print(f"Hausdorff vs GT (lat. units):  Mean = {df_res['hausdorff_vs_gt'].mean():.3f}, Median = {df_res['hausdorff_vs_gt'].median():.3f}")
    print(f"RMS vs Model Path:             Mean = {df_res['rms_vs_model'].mean():.3f}, Median = {df_res['rms_vs_model'].median():.3f}")
    print(f"Hausdorff vs Model Path:       Mean = {df_res['hausdorff_vs_model'].mean():.3f}, Median = {df_res['hausdorff_vs_model'].median():.3f}")
    print(f"Beta_0 Exact Match Rate:       {df_res['beta0_match'].mean() * 100:.1f}%")
    print(f"Beta_1 Exact Match Rate:       {df_res['beta1_match'].mean() * 100:.1f}%")
    print(f"Chosen K (Harmonics):          Mean = {df_res['chosen_K'].mean():.1f}, Median = {df_res['chosen_K'].median():.1f}")
    print("=================================================================\n")
    
    # Plot K vs Error vs Beta1
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Sample Index (Held-out)')
    ax1.set_ylabel('RMS Error vs GT (lattice units)', color='tab:blue')
    ax1.plot(df_res['sample_idx'], df_res['rms_vs_gt'], color='tab:blue', marker='o', linewidth=1.5, label='RMS vs GT')
    ax1.axhline(0.25, color='gray', linestyle='--', alpha=0.7, label='Tolerance eps=0.25')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Chosen Harmonic Count K', color='tab:purple')
    ax2.plot(df_res['sample_idx'], df_res['chosen_K'], color='tab:purple', marker='s', linestyle='--', linewidth=1.5, label='Chosen K')
    
    for idx, row in df_res.iterrows():
        color = 'green' if row['beta1_match'] else 'red'
        ax1.scatter(row['sample_idx'], row['rms_vs_gt'], color=color, s=70, zorder=5)
        
    plt.title(f"Validation: K vs Error vs Topological Match across N={N} Stratified Held-out Samples")
    fig.tight_layout()
    plot_out = os.path.join("reports", "k_vs_error_vs_beta1.png")
    plt.savefig(plot_out, dpi=200)
    plt.close()
    print(f"Saved plot to {plot_out}")

if __name__ == "__main__":
    run_validation()
