import os
import json
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from config import Config
from model import KolamModel
from dataset import KolamPathDataset, denormalize
from topology import render_path_to_binary, compute_betti_numbers

def chamfer_distance(pts1, pts2):
    """Symmetric nearest-point Chamfer distance between two point sets in lattice units."""
    d = cdist(pts1, pts2)
    return float(0.5 * (np.mean(np.min(d, axis=1)) + np.mean(np.min(d, axis=0))))

def hausdorff_distance(pts1, pts2):
    """Symmetric Hausdorff distance between two point sets in lattice units."""
    d = cdist(pts1, pts2)
    return float(max(np.max(np.min(d, axis=1)), np.max(np.min(d, axis=0))))

def fast_dtw(seq1, seq2):
    """Fast DTW calculation."""
    n, m = len(seq1), len(seq2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])

def snap_deltas_lattice(delta_logical):
    """
    Round each predicted move * 2 to nearest integer vector with norm > 0.
    All GT moves lie on a half-lattice (multiples of 0.5).
    """
    snapped = np.round(delta_logical * 2.0) / 2.0
    norm = np.linalg.norm(snapped)
    if norm == 0:
        dom = np.argmax(np.abs(delta_logical))
        snapped[dom] = 0.5 * np.sign(delta_logical[dom]) if delta_logical[dom] != 0 else 0.5
    return snapped

@torch.no_grad()
def rollout_inference(model, img_tensor, config, mode='raw'):
    """
    Greedy rollout from fixed start (1.0, 0.0).
    Stops at EOS (eos_logit > 0).
    Modes:
      'raw': standard autoregressive delta accumulation
      'snapped': each step delta is snapped to nearest half-lattice vector
    Returns:
      coords_logical: (T, 2) numpy array in un-normalized logical coordinates [-18.5, 18.5]
    """
    model.eval()
    device = config.device
    memory = model.encoder(img_tensor)
    
    # Fixed logical start: [1.0, 0.0]
    # In normalized space [-1, 1]:
    c_range = config.coord_max - config.coord_min # 37.0
    start_norm_x = (1.0 - config.coord_min) / c_range * 2.0 - 1.0
    start_norm_y = (0.0 - config.coord_min) / c_range * 2.0 - 1.0
    
    curr_coords = torch.tensor([[[start_norm_x, start_norm_y]]], dtype=torch.float32, device=device)
    curr_pen = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Tracking logical trajectory
    logical_path = [[1.0, 0.0]]
    
    for t in range(config.max_seq_len - 1):
        delta_coords, eos_logits = model.decoder(memory, curr_coords, curr_pen)
        next_delta = delta_coords[:, -1:, :]
        next_eos = eos_logits[:, -1, 0].item()
        
        # Stop at EOS if generated at least 50 steps
        if next_eos > 0.0 and t >= 50:
            break
            
        nd = next_delta[0, 0].cpu().numpy()
        # Convert delta from normalized [-1, 1] scale to logical scale [-18.5, 18.5]
        # delta_logical = delta_norm * (c_range / 2.0)
        delta_logical = nd * (c_range / 2.0)
        
        if mode == 'snapped':
            delta_logical = snap_deltas_lattice(delta_logical)
            # Re-convert snapped delta back to normalized for next step input
            nd = delta_logical / (c_range / 2.0)
            next_delta = torch.tensor(nd, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
        next_logical = [logical_path[-1][0] + delta_logical[0], logical_path[-1][1] + delta_logical[1]]
        logical_path.append(next_logical)
        
        next_coord = curr_coords[:, -1:, :] + next_delta
        next_pen = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        curr_coords = torch.cat([curr_coords, next_coord], dim=1)
        curr_pen = torch.cat([curr_pen, next_pen], dim=1)
        
    return np.array(logical_path, dtype=np.float32)

def evaluate_held_out_40(model, config, splits_path="splits.json"):
    """
    Evaluates 40 held-out images against pair-1 ground-truth path (725 points).
    Computes:
      - Chamfer (lattice units)
      - Hausdorff (lattice units)
      - DTW
      - Closure distance ||p_end - p_start|| (lattice units)
      - Beta1 cycle count (Euler-Poincaré)
    for both 'raw' and 'snapped' rollout modes.
    """
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        
    held_images = splits['held_images']
    held_csvs = splits['held_csvs']
    N = len(held_images)
    
    # Ground truth pair-1 path (725 points)
    df_gt = pd.read_csv(held_csvs[0])
    if not {'x', 'y'}.issubset(set(df_gt.columns)) and len(df_gt.columns) >= 2:
        df_gt.columns = ['x', 'y'] + list(df_gt.columns[2:])
    gt_pts = df_gt[['x', 'y']].values[:725].astype(np.float32)
    
    # GT rasterization topology
    gt_bin = render_path_to_binary(gt_pts[:, 0], gt_pts[:, 1], img_size=128)
    gt_b0, gt_b1 = compute_betti_numbers(gt_bin)
    
    val_dataset = KolamPathDataset(held_images, held_csvs, config)
    
    raw_metrics = dict(chamfer=[], hausdorff=[], dtw=[], closure=[], beta1=[])
    snapped_metrics = dict(chamfer=[], hausdorff=[], dtw=[], closure=[], beta1=[])
    
    for i in range(N):
        sample = val_dataset[i]
        img_tensor = sample['image'].unsqueeze(0).to(config.device)
        
        # 1. Raw rollout
        p_raw = rollout_inference(model, img_tensor, config, mode='raw')
        ch_r = chamfer_distance(p_raw, gt_pts)
        hd_r = hausdorff_distance(p_raw, gt_pts)
        dtw_r = fast_dtw(p_raw, gt_pts)
        clos_r = float(np.linalg.norm(p_raw[-1] - p_raw[0]))
        r_bin = render_path_to_binary(p_raw[:, 0], p_raw[:, 1], img_size=128)
        _, r_b1 = compute_betti_numbers(r_bin)
        
        raw_metrics['chamfer'].append(ch_r)
        raw_metrics['hausdorff'].append(hd_r)
        raw_metrics['dtw'].append(dtw_r)
        raw_metrics['closure'].append(clos_r)
        raw_metrics['beta1'].append(r_b1)
        
        # 2. Snapped rollout
        p_snap = rollout_inference(model, img_tensor, config, mode='snapped')
        ch_s = chamfer_distance(p_snap, gt_pts)
        hd_s = hausdorff_distance(p_snap, gt_pts)
        dtw_s = fast_dtw(p_snap, gt_pts)
        clos_s = float(np.linalg.norm(p_snap[-1] - p_snap[0]))
        s_bin = render_path_to_binary(p_snap[:, 0], p_snap[:, 1], img_size=128)
        _, s_b1 = compute_betti_numbers(s_bin)
        
        snapped_metrics['chamfer'].append(ch_s)
        snapped_metrics['hausdorff'].append(hd_s)
        snapped_metrics['dtw'].append(dtw_s)
        snapped_metrics['closure'].append(clos_s)
        snapped_metrics['beta1'].append(s_b1)
        
    summary = {
        'gt_beta1': gt_b1,
        'raw': {
            'chamfer_mean': float(np.mean(raw_metrics['chamfer'])),
            'chamfer_median': float(np.median(raw_metrics['chamfer'])),
            'hausdorff_mean': float(np.mean(raw_metrics['hausdorff'])),
            'hausdorff_median': float(np.median(raw_metrics['hausdorff'])),
            'dtw_mean': float(np.mean(raw_metrics['dtw'])),
            'closure_mean': float(np.mean(raw_metrics['closure'])),
            'beta1_mean': float(np.mean(raw_metrics['beta1'])),
        },
        'snapped': {
            'chamfer_mean': float(np.mean(snapped_metrics['chamfer'])),
            'chamfer_median': float(np.median(snapped_metrics['chamfer'])),
            'hausdorff_mean': float(np.mean(snapped_metrics['hausdorff'])),
            'hausdorff_median': float(np.median(snapped_metrics['hausdorff'])),
            'dtw_mean': float(np.mean(snapped_metrics['dtw'])),
            'closure_mean': float(np.mean(snapped_metrics['closure'])),
            'beta1_mean': float(np.mean(snapped_metrics['beta1'])),
        }
    }
    return summary
