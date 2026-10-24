import os
import json
import torch
import numpy as np
import pandas as pd
from config import Config
from model import KolamModel
from dataset import KolamPathDataset, denormalize

def fast_dtw(seq1, seq2):
    """Fast DTW calculation between two sequences (T1, 2) and (T2, 2)."""
    n, m = len(seq1), len(seq2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])

def evaluate_subset(model, dataset, config, n_samples=30):
    """Evaluates greedy rollout (TF=0, true start) DTW on n_samples from dataset."""
    model.eval()
    dtw_list = []
    total = min(len(dataset), n_samples)
    
    with torch.no_grad():
        for i in range(total):
            sample = dataset[i]
            img = sample['image'].unsqueeze(0).to(config.device)
            coords = sample['target_coords'].unsqueeze(0).to(config.device)
            # Greedy rollout with true start coordinate
            start_coords = coords[:, 0:1, :]
            
            memory = model.encoder(img)
            pred_coords, _ = model.decoder.autoregressive_generate(
                memory, coords.shape[1], start_coords=start_coords
            )
            
            pred_np = pred_coords[0].cpu().numpy()
            gt_np = sample['target_coords'].numpy()
            
            seq_len = int(sample['seq_len'].item()) + 1
            dtw_val = fast_dtw(pred_np[:seq_len], gt_np[:seq_len])
            dtw_list.append(dtw_val)
            
    return {
        'mean': float(np.mean(dtw_list)),
        'median': float(np.median(dtw_list)),
        'values': dtw_list
    }

def run_eval_multi(model=None, config=None, splits_path="splits.json", ckpt_path=None, expected_hash=None):
    """
    Evaluates 30 train samples + 30 held-out samples with greedy rollout (TF=0, true start).
    Asserts held_out_hash matches splits.json and checkpoint.
    """
    if config is None:
        config = Config()
        
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Splits file {splits_path} not found.")
        
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        
    splits_hash = splits.get('held_out_hash')
    if expected_hash is not None and splits_hash is not None:
        assert expected_hash == splits_hash, f"Split hash mismatch! Expected {expected_hash}, got {splits_hash}"
        
    if model is None and ckpt_path is not None:
        model = KolamModel(config).to(config.device)
        ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
        if 'held_out_hash' in ckpt:
            assert ckpt['held_out_hash'] == splits_hash, f"Checkpoint hash mismatch! Checkpoint: {ckpt['held_out_hash']}, Splits: {splits_hash}"
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        
    train_dataset = KolamPathDataset(splits['train_images'], splits['train_csvs'], config)
    held_dataset = KolamPathDataset(splits['held_images'], splits['held_csvs'], config)
    
    train_metrics = evaluate_subset(model, train_dataset, config, n_samples=30)
    held_metrics = evaluate_subset(model, held_dataset, config, n_samples=30)
    
    print(f"  [eval_multi] Train (30): Mean DTW={train_metrics['mean']:.4f}, Median={train_metrics['median']:.4f}", flush=True)
    print(f"  [eval_multi] Held  (30): Mean DTW={held_metrics['mean']:.4f}, Median={held_metrics['median']:.4f}", flush=True)
    
    return {
        'train_mean': train_metrics['mean'],
        'train_median': train_metrics['median'],
        'held_mean': held_metrics['mean'],
        'held_median': held_metrics['median'],
    }

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/checkpoint_epoch_015.pt"
    run_eval_multi(ckpt_path=ckpt)
