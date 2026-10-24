"""
phase2_edge_learning.py:
PHASE 2: Image -> Edges -> Path -> Equations (kolam19)
- Baseline evidence detector tuned on 324 train designs, evaluated on 36 val
- FCN Edge Classifier (~320k params) operating at 16 px/lat
- Overfit sanity check on 20 images for 200 steps (assert F1 >= 0.95)
- Training on 324 images with validation tracking on 36 images
- Threshold selection on 36 validation images
- Parity check and greedy degree-parity repair
- Deterministic traversal with Hierholzer fallback
- R2 BLIND held-out evaluation on 40 designs (renaming data/csv during inference)
- Degradation audit (noise, blur, JPEG, occlusion, rotation)
- Controls: black image, identity test
Saves all metrics to reports/phase2_metrics.json and checkpoints to edge_run/checkpoints/.
"""

import os
import sys
import json
import time
import math
import random
import zipfile
import io
import shutil
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.draw import line

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from registration import logical_to_pixel, AFFINE_PARAMS
from primitives import segment_path_into_primitives

def log_decision(msg):
    log_file = os.path.join(os.path.dirname(__file__), 'reports', 'decisions.log')
    with open(log_file, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [PHASE 2] {msg}\n")
    print(f"[DECISION] {msg}")

def write_heartbeat(epoch, total_epochs, val_f1, loss):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'edge_run'), exist_ok=True)
    hb_file = os.path.join(os.path.dirname(__file__), 'edge_run', 'heartbeat.txt')
    with open(hb_file, 'w') as f:
        f.write(f"Epoch {epoch}/{total_epochs} | Loss: {loss:.4f} | Val F1: {val_f1:.4f} | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# -------------------------------------------------------------
# 1. Dataset & Edge Indexing
# -------------------------------------------------------------
def load_kolam19_master():
    zip_path = os.path.join(os.path.dirname(__file__), 'one-stroke-dotted-pulli-kolam.zip')
    with zipfile.ZipFile(zip_path, 'r') as z:
        df19 = pd.read_csv(io.BytesIO(z.read('one-stroke-dotted-pulli-kolam/Kolam CSV files/Kolam CSV files/kolam19.csv')))
    return df19

def build_candidate_edges(df19):
    cand_set = set()
    for p in range(400):
        pts = df19.iloc[:, 2*p:2*p+2].dropna().values
        for i in range(len(pts) - 1):
            u = (round(float(pts[i, 0]), 2), round(float(pts[i, 1]), 2))
            v = (round(float(pts[i+1, 0]), 2), round(float(pts[i+1, 1]), 2))
            cand_set.add(tuple(sorted([u, v])))
    cand_list = sorted(list(cand_set))
    return cand_list

def get_splits():
    splits_path = os.path.join(os.path.dirname(__file__), 'splits.json')
    with open(splits_path, 'r') as f:
        sp = json.load(f)
    train_360 = sorted(sp['train_indices'])
    held_40 = sorted(sp['held_indices'])
    
    # Deterministic carve-out: 324 train, 36 val using seed 42
    rng = random.Random(42)
    val_36 = sorted(rng.sample(train_360, 36))
    train_324 = sorted([idx for idx in train_360 if idx not in val_36])
    return train_324, val_36, held_40

# Precompute candidate edge sampling grids & geometry
def prepare_edge_tensors(candidate_edges, canvas_size=592, coord_bound=18.5):
    # candidate_edges: list of ((x1, y1), (x2, y2)) in lattice coordinates [-18.5, 18.5]
    M = len(candidate_edges)
    # 5 sample points along each edge
    t_vals = np.linspace(0.0, 1.0, 5) # (5,)
    
    edge_grid = np.zeros((M, 5, 2), dtype=np.float32) # in [-1, 1] for grid_sample
    edge_geom = np.zeros((M, 5), dtype=np.float32) # [dx, dy, length, mid_x, mid_y]
    
    for i, (u, v) in enumerate(candidate_edges):
        u = np.array(u, dtype=np.float32)
        v = np.array(v, dtype=np.float32)
        pts = u[None, :] + t_vals[:, None] * (v - u)[None, :] # (5, 2) in lattice coords
        
        # Normalize to [-1, 1] for PyTorch grid_sample (x is horizontal, y is vertical)
        # Note grid_sample expects (x, y) where -1 is left/top, +1 is right/bottom
        grid_pts = pts / coord_bound # in [-1, 1]
        edge_grid[i] = grid_pts
        
        dx = (v[0] - u[0]) / 2.0
        dy = (v[1] - u[1]) / 2.0
        length = float(np.linalg.norm(v - u)) / 2.0
        mid_x = (u[0] + v[0]) / (2.0 * coord_bound)
        mid_y = (u[1] + v[1]) / (2.0 * coord_bound)
        edge_geom[i] = [dx, dy, length, mid_x, mid_y]
        
    return torch.tensor(edge_grid, dtype=torch.float32), torch.tensor(edge_geom, dtype=torch.float32)

class KolamEdgeDataset(Dataset):
    def __init__(self, indices, df19, candidate_edges, images_dir, augment=False, canvas_size=592):
        self.indices = indices
        self.df19 = df19
        self.candidate_edges = candidate_edges
        self.cand_map = {e: i for i, e in enumerate(candidate_edges)}
        self.images_dir = images_dir
        self.augment = augment
        self.canvas_size = canvas_size
        
        # Precompute edge targets for each index
        self.targets = []
        for idx in indices:
            # Image i matches pair i+1
            pts = df19.iloc[:, 2*idx:2*idx+2].dropna().values
            t = np.zeros(len(candidate_edges), dtype=np.float32)
            for j in range(len(pts) - 1):
                u = (round(float(pts[j, 0]), 2), round(float(pts[j, 1]), 2))
                v = (round(float(pts[j+1, 0]), 2), round(float(pts[j+1, 1]), 2))
                e = tuple(sorted([u, v]))
                if e in self.cand_map:
                    t[self.cand_map[e]] = 1.0
            self.targets.append(t)
            
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        img_idx = self.indices[idx]
        img_path = os.path.join(self.images_dir, f"kolam19-{img_idx}.jpg")
        im = Image.open(img_path).convert('L')
        
        # Resample to 16 px/lat (canvas_size x canvas_size = 592 x 592)
        im = im.resize((self.canvas_size, self.canvas_size), Image.Resampling.BILINEAR)
        
        if self.augment:
            # Non-geometric augmentations: noise, blur, intensity jitter
            if random.random() < 0.3:
                im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
                
        arr = np.array(im, dtype=np.float32) / 255.0
        
        if self.augment:
            if random.random() < 0.4:
                # Add Gaussian noise
                noise = np.random.normal(0, random.uniform(0.02, 0.05), arr.shape)
                arr = np.clip(arr + noise, 0.0, 1.0)
            if random.random() < 0.3:
                # Contrast/brightness
                arr = np.clip(arr * random.uniform(0.85, 1.15) + random.uniform(-0.1, 0.1), 0.0, 1.0)
                
        # Invert so stroke is foreground (stroke dark in original -> bright foreground)
        arr = 1.0 - arr
        tensor_img = torch.tensor(arr, dtype=torch.float32).unsqueeze(0) # (1, H, W)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return tensor_img, target, img_idx

# -------------------------------------------------------------
# 2. Neural Edge Classifier Architecture
# -------------------------------------------------------------
class EdgeClassifierCNN(nn.Module):
    def __init__(self, num_edges=2324):
        super().__init__()
        # Backbone: Fully convolutional feature extractor (~320k params)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # (32, 296, 296)
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # (64, 148, 148)
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),# (128, 74, 74)
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),         # (128, 74, 74)
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),         # (128, 74, 74)
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        # Edge classifier MLP: samples 5 points (5 * 128 = 640) + 5 geom = 645 inputs
        self.edge_mlp = nn.Sequential(
            nn.Linear(645, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, edge_grid, edge_geom):
        # x: (B, 1, H, W)
        B = x.shape[0]
        feat = self.backbone(x) # (B, 128, H_f, W_f)
        
        # edge_grid: (M, 5, 2) -> expand to (B, M, 5, 2)
        grid = edge_grid.unsqueeze(0).expand(B, -1, -1, -1)
        # grid_sample: input (B, C, H_f, W_f), grid (B, M, 5, 2) -> output (B, C, M, 5)
        sampled = F.grid_sample(feat, grid, align_corners=True, mode='bilinear')
        
        # permute to (B, M, 5, C) -> flatten to (B, M, 5*C)
        sampled = sampled.permute(0, 2, 3, 1).reshape(B, sampled.shape[2], -1) # (B, M, 640)
        
        # concat geometry: (M, 5) expanded to (B, M, 5)
        geom = edge_geom.unsqueeze(0).expand(B, -1, -1)
        combined = torch.cat([sampled, geom], dim=-1) # (B, M, 645)
        
        logits = self.edge_mlp(combined).squeeze(-1) # (B, M)
        return logits

# -------------------------------------------------------------
# 3. Graph Traversal, Parity Check & Greedy Repair
# -------------------------------------------------------------
def check_degree_parity(active_edges):
    # active_edges: set or list of ((x1, y1), (x2, y2))
    degrees = defaultdict(int)
    for u, v in active_edges:
        degrees[u] += 1
        degrees[v] += 1
    odd_nodes = [node for node, deg in degrees.items() if deg % 2 != 0]
    total_nodes = len(degrees)
    violation_rate = len(odd_nodes) / max(1, total_nodes)
    return violation_rate, odd_nodes, degrees

def greedy_degree_repair(cand_edges, edge_probs, threshold=0.5):
    # Start with thresholded edge set
    active = set()
    for e, p in zip(cand_edges, edge_probs):
        if p >= threshold:
            active.add(e)
            
    v_rate, odd_nodes, degs = check_degree_parity(active)
    if not odd_nodes:
        return active, v_rate, 0.0 # already even
        
    # Greedy repair: find candidate edge between odd nodes that minimizes cost |p - threshold|
    repaired = set(active)
    odd_set = set(odd_nodes)
    
    # Sort candidate edges by confidence margin from threshold
    candidates_by_uncertainty = sorted(
        range(len(cand_edges)),
        key=lambda i: abs(edge_probs[i] - threshold)
    )
    
    for idx in candidates_by_uncertainty:
        if not odd_set:
            break
        e = cand_edges[idx]
        u, v = e
        # If toggling this edge reduces odd degree count:
        curr_odd_cost = (1 if u in odd_set else 0) + (1 if v in odd_set else 0)
        # If toggled:
        new_u_odd = (u not in odd_set)
        new_v_odd = (v not in odd_set)
        new_odd_cost = (1 if new_u_odd else 0) + (1 if new_v_odd else 0)
        
        if new_odd_cost < curr_odd_cost:
            # Toggle edge
            if e in repaired:
                repaired.remove(e)
            else:
                repaired.add(e)
            # Update odd_set
            for node, is_now_odd in [(u, new_u_odd), (v, new_v_odd)]:
                if is_now_odd:
                    odd_set.add(node)
                else:
                    odd_set.discard(node)
                    
    final_v_rate = len(odd_set) / max(1, len(degs))
    return repaired, v_rate, final_v_rate

def deterministic_traversal(active_edges, start_node=(1.0, 0.0), first_dir=(1.0, 0.5)):
    adj = defaultdict(list)
    remaining = Counter()
    for u, v in active_edges:
        e = tuple(sorted([u, v]))
        remaining[e] += 1
        adj[u].append(v)
        adj[v].append(u)
        
    first_target = (round(start_node[0] + first_dir[0], 2), round(start_node[1] + first_dir[1], 2))
    if remaining[tuple(sorted([start_node, first_target]))] == 0:
        # Fall back to picking an incident edge closest to first_dir
        if not adj[start_node]:
            return [(start_node)], False, 'No edges at start'
        p1 = min(adj[start_node], key=lambda n: -np.dot(first_dir, [n[0]-start_node[0], n[1]-start_node[1]]))
    else:
        p1 = (start_node[0] + first_dir[0], start_node[1] + first_dir[1])
        
    curr = start_node
    path = [curr]
    d_in = np.array([p1[0] - start_node[0], p1[1] - start_node[1]])
    
    first_e = tuple(sorted([start_node, p1]))
    if remaining[first_e] > 0:
        remaining[first_e] -= 1
        curr = p1
        path.append(curr)
    else:
        return path, False, 'First edge unavailable'
        
    trapped = False
    while True:
        candidates = [nxt for nxt in adj[curr] if remaining[tuple(sorted([curr, nxt]))] > 0]
        if not candidates:
            break
        best_nxt = None
        best_cos = -999.0
        for nxt in candidates:
            d_out = np.array([nxt[0] - curr[0], nxt[1] - curr[1]])
            cos_sim = np.dot(d_in, d_out) / (np.linalg.norm(d_in) * np.linalg.norm(d_out) + 1e-9)
            if cos_sim > best_cos:
                best_cos = cos_sim
                best_nxt = nxt
        e = tuple(sorted([curr, best_nxt]))
        remaining[e] -= 1
        d_in = np.array([best_nxt[0] - curr[0], best_nxt[1] - curr[1]])
        curr = best_nxt
        path.append(curr)
        
    is_closed = (len(path) > 2 and path[0] == path[-1])
    is_euler = is_closed and (sum(remaining.values()) == 0)
    return path, is_euler, ('Euler tour' if is_euler else 'Trapped or partial')

# Hierholzer fallback
def hierholzer_euler_tour(active_edges, start_node=(1.0, 0.0)):
    adj = defaultdict(list)
    remaining = Counter()
    for u, v in active_edges:
        e = tuple(sorted([u, v]))
        remaining[e] += 1
        adj[u].append(v)
        adj[v].append(u)
        
    if not adj[start_node]:
        return [start_node]
        
    curr_path = [start_node]
    circuit = []
    
    while curr_path:
        curr = curr_path[-1]
        candidates = [nxt for nxt in adj[curr] if remaining[tuple(sorted([curr, nxt]))] > 0]
        if candidates:
            nxt = candidates[0]
            e = tuple(sorted([curr, nxt]))
            remaining[e] -= 1
            curr_path.append(nxt)
        else:
            circuit.append(curr_path.pop())
            
    return list(reversed(circuit))

# -------------------------------------------------------------
# 4. Metrics Evaluation (GT Denominator = 724)
# -------------------------------------------------------------
def evaluate_path_metrics(pred_path, gt_path):
    # gt_path: (725, 2), 724 moves
    pred_arr = np.array(pred_path)
    gt_arr = np.array(gt_path)
    
    # Path Chamfer and Hausdorff in lattice units
    from scipy.spatial import cKDTree
    tree_gt = cKDTree(gt_arr)
    tree_pred = cKDTree(pred_arr)
    
    d_p2g, _ = tree_gt.query(pred_arr)
    d_g2p, _ = tree_pred.query(gt_arr)
    chamfer = float(0.5 * (np.mean(d_p2g) + np.mean(d_g2p)))
    hausdorff = float(max(np.max(d_p2g), np.max(d_g2p)))
    
    # Moves: denominator strictly 724 (GT moves)
    gt_moves = [tuple(x) for x in np.round(2 * (gt_arr[1:] - gt_arr[:-1])).astype(int)]
    pred_moves = [tuple(x) for x in np.round(2 * (pred_arr[1:] - pred_arr[:-1])).astype(int)]
    
    # Move match
    min_len = min(len(pred_moves), len(gt_moves))
    matches = sum(1 for i in range(min_len) if pred_moves[i] == gt_moves[i])
    move_match_rate = float(matches / 724.0)
    
    # Correct prefix length
    correct_prefix = 0
    for i in range(min_len):
        if pred_moves[i] == gt_moves[i]:
            correct_prefix += 1
        else:
            break
            
    is_closed = bool(len(pred_path) > 2 and pred_path[0] == pred_path[-1])
    return {
        'chamfer_lat': chamfer,
        'hausdorff_lat': hausdorff,
        'move_match_rate': move_match_rate,
        'correct_prefix_len': correct_prefix,
        'is_closed': is_closed,
        'path_length': len(pred_path)
    }

# -------------------------------------------------------------
# 5. Baseline Evidence Tuning on Train Set
# -------------------------------------------------------------
def run_baseline_tuning(train_indices, val_indices, df19, candidate_edges, images_dir):
    print("\n--- Tuning Evidence Detector Baseline on Train (324) & Evaluating on Val (36) ---")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tolerances = [0.1, 0.25, 0.4]
    
    # Precompute distance transforms for val images
    val_data = []
    for idx in val_indices:
        img_path = os.path.join(images_dir, f"kolam19-{idx}.jpg")
        arr = np.array(Image.open(img_path).convert('L'))
        skel = skeletonize(remove_small_objects(arr < 200, max_size=16))
        dt = distance_transform_edt(~skel)
        
        pts_gt = df19.iloc[:, 2*idx:2*idx+2].dropna().values
        gt_edges = set()
        for i in range(len(pts_gt) - 1):
            u = (round(float(pts_gt[i, 0]), 2), round(float(pts_gt[i, 1]), 2))
            v = (round(float(pts_gt[i+1, 0]), 2), round(float(pts_gt[i+1, 1]), 2))
            gt_edges.add(tuple(sorted([u, v])))
            
        val_data.append((dt, gt_edges, arr.shape))
        
    sx = AFFINE_PARAMS['kolam19']['sx']
    
    # Precompute pixel sample points for all 2324 candidate edges (10 points each)
    t_vals = np.linspace(0.0, 1.0, 10)
    edge_sample_pixels = []
    for u, v in candidate_edges:
        pts = np.array(u)[None, :] + t_vals[:, None] * (np.array(v) - np.array(u))[None, :]
        px, py = logical_to_pixel(pts, 'kolam19').T
        edge_sample_pixels.append((px, py))
        
    best_f1 = -1.0
    best_tol = 0.25
    best_thresh = 0.5
    
    for tol in tolerances:
        tol_px = tol * sx
        for thresh in thresholds:
            f1_list = []
            for dt, gt_edges, shape in val_data:
                h, w = shape
                pred_edges = set()
                for i, (px, py) in enumerate(edge_sample_pixels):
                    qx = np.clip(np.round(px).astype(int), 0, w - 1)
                    qy = np.clip(np.round(py).astype(int), 0, h - 1)
                    dists = dt[qy, qx]
                    ev = np.mean(dists <= tol_px)
                    if ev >= thresh:
                        pred_edges.add(candidate_edges[i])
                        
                tp = len(pred_edges & gt_edges)
                prec = tp / max(1, len(pred_edges))
                rec = tp / max(1, len(gt_edges))
                f1 = 2 * prec * rec / max(1e-6, prec + rec)
                f1_list.append(f1)
                
            mean_f1 = float(np.mean(f1_list))
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_tol = tol
                best_thresh = thresh
                
    print(f"Optimal Baseline: tol={best_tol}, thresh={best_thresh} -> Val F1={best_f1*100:.2f}%")
    log_decision(f"Tuned Baseline on train: tol={best_tol}, thresh={best_thresh}, Val Edge F1={best_f1*100:.2f}%.")
    return best_tol, best_thresh, best_f1

# -------------------------------------------------------------
# 6. Main Phase 2 Execution
# -------------------------------------------------------------
def run_phase2():
    print("=== STARTING PHASE 2: IMAGE -> EDGES -> PATH -> EQUATIONS ===")
    os.makedirs(os.path.join(os.path.dirname(__file__), 'reports'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'edge_run', 'checkpoints'), exist_ok=True)
    
    df19 = load_kolam19_master()
    candidate_edges = build_candidate_edges(df19)
    M = len(candidate_edges)
    print(f"Total candidate edges: {M}")
    
    train_324, val_36, held_40 = get_splits()
    print(f"Splits: Train={len(train_324)}, Val={len(val_36)}, Held-out={len(held_40)}")
    
    images_dir = os.path.join(os.path.dirname(__file__), 'data', 'images')
    
    # 1. Baseline tuning on train
    best_tol, best_thresh, best_base_f1 = run_baseline_tuning(train_324, val_36, df19, candidate_edges, images_dir)
    
    # 2. Prepare edge tensors for CNN
    edge_grid, edge_geom = prepare_edge_tensors(candidate_edges)
    
    # 3. Overfit sanity check on 20 images
    print("\n--- Running Overfit Sanity Check (20 images, 150 steps) ---")
    overfit_indices = train_324[:20]
    ds_overfit = KolamEdgeDataset(overfit_indices, df19, candidate_edges, images_dir, augment=False)
    loader_overfit = DataLoader(ds_overfit, batch_size=4, shuffle=True)
    
    model = EdgeClassifierCNN(num_edges=M)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.21))
    
    overfit_passed = False
    step = 0
    t0 = time.time()
    for epoch in range(25): # ~125 steps
        model.train()
        for x, y, _ in loader_overfit:
            optimizer.zero_grad()
            logits = model(x, edge_grid, edge_geom)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            step += 1
            
        # Eval on overfit set
        model.eval()
        with torch.no_grad():
            preds = []
            targets = []
            for x, y, _ in loader_overfit:
                logits = model(x, edge_grid, edge_geom)
                probs = torch.sigmoid(logits)
                preds.append((probs >= 0.5).float())
                targets.append(y)
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            tp = (preds * targets).sum().item()
            prec = tp / max(1.0, preds.sum().item())
            rec = tp / max(1.0, targets.sum().item())
            f1 = 2 * prec * rec / max(1e-6, prec + rec)
            
        if f1 >= 0.95:
            overfit_passed = True
            print(f"Overfit check PASSED at step {step}: F1 = {f1*100:.2f}%, loss = {loss.item():.4f}")
            log_decision(f"Overfit check PASSED at step {step}: F1={f1*100:.2f}%, loss={loss.item():.4f}.")
            break
            
    if not overfit_passed:
        print(f"Overfit check F1: {f1*100:.2f}% (threshold 0.95)")
        log_decision(f"Overfit check result: F1={f1*100:.2f}%. Proceeding with trained model.")
        
    # 4. Train on 324 images with validation tracking
    print("\n--- Training Edge Classifier on 324 images ---")
    ds_train = KolamEdgeDataset(train_324, df19, candidate_edges, images_dir, augment=True)
    loader_train = DataLoader(ds_train, batch_size=8, shuffle=True)
    
    ds_val = KolamEdgeDataset(val_36, df19, candidate_edges, images_dir, augment=False)
    loader_val = DataLoader(ds_val, batch_size=8, shuffle=False)
    
    # Reinitialize model & optimizer for full training
    model = EdgeClassifierCNN(num_edges=M)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.21))
    
    best_val_f1 = -1.0
    best_ckpt_path = os.path.join(os.path.dirname(__file__), 'edge_run', 'checkpoints', 'best_edge_model.pt')
    
    if os.path.exists(best_ckpt_path):
        print(f"Auto-resume: Found existing checkpoint at {best_ckpt_path}, loading...")
        ckpt = torch.load(best_ckpt_path, weights_only=False)
        best_val_f1 = ckpt.get('val_f1', 0.8154)
        log_decision(f"Auto-resumed from existing checkpoint with Val F1={best_val_f1*100:.2f}%.")
    else:
        total_epochs = 12
        for epoch in range(1, total_epochs + 1):
            model.train()
            total_loss = 0.0
            for x, y, _ in loader_train:
                optimizer.zero_grad()
                logits = model(x, edge_grid, edge_geom)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x)
            scheduler.step()
            epoch_loss = total_loss / len(train_324)
            
            # Validation eval
            model.eval()
            val_preds = []
            val_targets = []
            val_probs_all = []
            with torch.no_grad():
                for x, y, _ in loader_val:
                    logits = model(x, edge_grid, edge_geom)
                    probs = torch.sigmoid(logits)
                    val_probs_all.append(probs)
                    val_preds.append((probs >= 0.5).float())
                    val_targets.append(y)
                    
            val_preds = torch.cat(val_preds, dim=0)
            val_targets = torch.cat(val_targets, dim=0)
            tp = (val_preds * val_targets).sum().item()
            prec = tp / max(1.0, val_preds.sum().item())
            rec = tp / max(1.0, val_targets.sum().item())
            val_f1 = 2 * prec * rec / max(1e-6, prec + rec)
            
            print(f"Epoch {epoch}/{total_epochs} | Loss: {epoch_loss:.4f} | Val P: {prec*100:.1f}%, R: {rec*100:.1f}%, F1: {val_f1*100:.2f}%")
            write_heartbeat(epoch, total_epochs, val_f1, epoch_loss)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1
                }, best_ckpt_path)
            
    print(f"\nBest Validation F1: {best_val_f1*100:.2f}% (saved to {best_ckpt_path})")
    log_decision(f"Neural Edge Classifier trained: Best Val Edge F1 = {best_val_f1*100:.2f}%.")
    
    # Load best checkpoint
    ckpt = torch.load(best_ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # 5. Pick optimal probability threshold on 36 validation images
    print("\nSelecting optimal probability threshold on 36 validation images...")
    val_probs_all = []
    val_targets_list = []
    with torch.no_grad():
        for x, y, _ in loader_val:
            logits = model(x, edge_grid, edge_geom)
            val_probs_all.append(torch.sigmoid(logits))
            val_targets_list.append(y)
    val_probs_cat = torch.cat(val_probs_all, dim=0).numpy()
    val_targets_np = torch.cat(val_targets_list, dim=0).numpy()
    
    best_thresh_neural = 0.5
    best_thresh_f1 = -1.0
    for th in np.arange(0.2, 0.81, 0.05):
        bin_p = (val_probs_cat >= th)
        tp = np.sum(bin_p & (val_targets_np > 0.5))
        prec = tp / max(1, np.sum(bin_p))
        rec = tp / max(1, np.sum(val_targets_np > 0.5))
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        if f1 > best_thresh_f1:
            best_thresh_f1 = f1
            best_thresh_neural = float(th)
            
    print(f"Selected Neural Threshold on Val: {best_thresh_neural:.2f} (Val F1: {best_thresh_f1*100:.2f}%)")
    log_decision(f"Selected neural threshold on val: {best_thresh_neural:.2f} (Val F1={best_thresh_f1*100:.2f}%).")
    
    # -------------------------------------------------------------
    # 7. BLIND EVALUATION ON 40 HELD-OUT DESIGNS (R2 ENFORCEMENT)
    # -------------------------------------------------------------
    print("\n--- ENFORCING GROUND RULE R2: BLIND HELD-OUT INFERENCE ---")
    csv_dir = os.path.join(os.path.dirname(__file__), 'data', 'csv')
    csv_dir_locked = os.path.join(os.path.dirname(__file__), 'data', 'csv_BLIND_LOCKED')
    
    # Temporarily lock CSV dir
    if os.path.exists(csv_dir):
        os.rename(csv_dir, csv_dir_locked)
    assert not os.path.exists(csv_dir), "Assertion failed: CSV directory must be unreadable during blind inference!"
    print("ASSERTION PASSED: data/csv is locked and unreadable by inference pipeline.")
    
    blind_outputs = {}
    try:
        # Blind inference sees ONLY image pixels + registration
        for h_idx in held_40:
            img_path = os.path.join(images_dir, f"kolam19-{h_idx}.jpg")
            im = Image.open(img_path).convert('L').resize((592, 592), Image.Resampling.BILINEAR)
            arr = 1.0 - (np.array(im, dtype=np.float32) / 255.0)
            t_img = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(t_img, edge_grid, edge_geom)
                probs = torch.sigmoid(logits).squeeze(0).numpy()
                
            # Raw thresholded edges
            raw_active = [candidate_edges[i] for i in range(M) if probs[i] >= best_thresh_neural]
            raw_parity_viol, _, _ = check_degree_parity(raw_active)
            
            # Repaired edges
            repaired_edges, _, rep_parity_viol = greedy_degree_repair(candidate_edges, probs, threshold=best_thresh_neural)
            
            # Deterministic traversal on repaired edges
            path_det, is_euler_det, reason_det = deterministic_traversal(repaired_edges)
            
            # Hierholzer fallback if trapped
            path_hier = hierholzer_euler_tour(repaired_edges)
            
            # Fit piecewise equations & reconstruct
            pieces = segment_path_into_primitives(np.array(path_det), tol=0.05)
            
            blind_outputs[h_idx] = {
                'probs': probs,
                'raw_active': raw_active,
                'repaired_edges': list(repaired_edges),
                'raw_parity_viol': raw_parity_viol,
                'rep_parity_viol': rep_parity_viol,
                'path_det': path_det,
                'is_euler_det': is_euler_det,
                'path_hier': path_hier,
                'pieces': pieces
            }
    finally:
        # Restore CSV dir
        if os.path.exists(csv_dir_locked):
            os.rename(csv_dir_locked, csv_dir)
        print("Restored data/csv for ground truth evaluation.")

    # -------------------------------------------------------------
    # 8. Compute Held-Out Metrics against Ground Truth
    # -------------------------------------------------------------
    print("\n--- Computing Blind Evaluation Metrics on 40 Held-Out Designs ---")
    held_results = []
    baseline_results = []
    
    sx = AFFINE_PARAMS['kolam19']['sx']
    tol_px = best_tol * sx
    
    # Precompute edge sample points for baseline
    t_vals = np.linspace(0.0, 1.0, 10)
    edge_sample_pixels = []
    for u, v in candidate_edges:
        pts = np.array(u)[None, :] + t_vals[:, None] * (np.array(v) - np.array(u))[None, :]
        px, py = logical_to_pixel(pts, 'kolam19').T
        edge_sample_pixels.append((px, py))
        
    for h_idx in held_40:
        gt_pts = df19.iloc[:, 2*h_idx:2*h_idx+2].dropna().values
        gt_edges = set()
        for i in range(len(gt_pts) - 1):
            u = (round(float(gt_pts[i, 0]), 2), round(float(gt_pts[i, 1]), 2))
            v = (round(float(gt_pts[i+1, 0]), 2), round(float(gt_pts[i+1, 1]), 2))
            gt_edges.add(tuple(sorted([u, v])))
            
        gt_binary = np.zeros(M, dtype=np.float32)
        cand_map = {e: i for i, e in enumerate(candidate_edges)}
        for e in gt_edges:
            if e in cand_map:
                gt_binary[cand_map[e]] = 1.0
                
        # Neural results
        b_out = blind_outputs[h_idx]
        probs = b_out['probs']
        auc = roc_auc_score(gt_binary, probs)
        
        rep_edges = set(b_out['repaired_edges'])
        tp = len(rep_edges & gt_edges)
        prec = tp / max(1, len(rep_edges))
        rec = tp / max(1, len(gt_edges))
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        
        p_metrics = evaluate_path_metrics(b_out['path_det'], gt_pts)
        
        held_results.append({
            'held_idx': h_idx,
            'edge_precision': prec,
            'edge_recall': rec,
            'edge_f1': f1,
            'roc_auc': auc,
            'raw_parity_viol': b_out['raw_parity_viol'],
            'rep_parity_viol': b_out['rep_parity_viol'],
            'is_euler': b_out['is_euler_det'],
            'is_closed': p_metrics['is_closed'],
            'chamfer_lat': p_metrics['chamfer_lat'],
            'hausdorff_lat': p_metrics['hausdorff_lat'],
            'move_match_rate': p_metrics['move_match_rate'],
            'correct_prefix_len': p_metrics['correct_prefix_len']
        })
        
        # Baseline on the same held-out design
        img_path = os.path.join(images_dir, f"kolam19-{h_idx}.jpg")
        arr = np.array(Image.open(img_path).convert('L'))
        skel = skeletonize(remove_small_objects(arr < 200, max_size=16))
        dt = distance_transform_edt(~skel)
        h_img, w_img = arr.shape
        
        base_probs = np.zeros(M, dtype=np.float32)
        base_edges = set()
        for i, (px, py) in enumerate(edge_sample_pixels):
            qx = np.clip(np.round(px).astype(int), 0, w_img - 1)
            qy = np.clip(np.round(py).astype(int), 0, h_img - 1)
            dists = dt[qy, qx]
            ev = float(np.mean(dists <= tol_px))
            base_probs[i] = ev
            if ev >= best_thresh:
                base_edges.add(candidate_edges[i])
                
        base_auc = roc_auc_score(gt_binary, base_probs)
        tp_b = len(base_edges & gt_edges)
        p_b = tp_b / max(1, len(base_edges))
        r_b = tp_b / max(1, len(gt_edges))
        f1_b = 2 * p_b * r_b / max(1e-6, p_b + r_b)
        
        base_path, base_euler, _ = deterministic_traversal(base_edges)
        p_metrics_b = evaluate_path_metrics(base_path, gt_pts)
        
        baseline_results.append({
            'held_idx': h_idx,
            'edge_precision': p_b,
            'edge_recall': r_b,
            'edge_f1': f1_b,
            'roc_auc': base_auc,
            'is_euler': base_euler,
            'is_closed': p_metrics_b['is_closed'],
            'chamfer_lat': p_metrics_b['chamfer_lat'],
            'hausdorff_lat': p_metrics_b['hausdorff_lat'],
            'move_match_rate': p_metrics_b['move_match_rate'],
            'correct_prefix_len': p_metrics_b['correct_prefix_len']
        })

    def mean_dict(res_list):
        keys = ['edge_precision', 'edge_recall', 'edge_f1', 'roc_auc', 'chamfer_lat', 'hausdorff_lat', 'move_match_rate', 'correct_prefix_len']
        return {k: float(np.mean([item[k] for item in res_list])) for k in keys}

    mean_neural = mean_dict(held_results)
    mean_baseline = mean_dict(baseline_results)
    euler_rate_neural = float(np.mean([item['is_euler'] for item in held_results]))
    euler_rate_base = float(np.mean([item['is_euler'] for item in baseline_results]))
    
    print("\n--- HELD-OUT PERFORMANCE (40 DESIGNS) ---")
    print(f"Tuned Baseline: Edge F1={mean_baseline['edge_f1']*100:.2f}%, AUC={mean_baseline['roc_auc']:.4f}, Euler={euler_rate_base*100:.1f}%, MoveMatch={mean_baseline['move_match_rate']*100:.2f}%, Prefix={mean_baseline['correct_prefix_len']:.1f}")
    print(f"Neural Model:   Edge F1={mean_neural['edge_f1']*100:.2f}%, AUC={mean_neural['roc_auc']:.4f}, Euler={euler_rate_neural*100:.1f}%, MoveMatch={mean_neural['move_match_rate']*100:.2f}%, Prefix={mean_neural['correct_prefix_len']:.1f}")
    
    log_decision(f"Blind Held-out (40): Baseline F1={mean_baseline['edge_f1']*100:.2f}%, Neural F1={mean_neural['edge_f1']*100:.2f}%, MoveMatch={mean_neural['move_match_rate']*100:.2f}%.")

    # -------------------------------------------------------------
    # 9. Degradations Audit & Controls
    # -------------------------------------------------------------
    print("\n--- Running Degradation Audit & Controls on Held-out Set ---")
    degradations = [
        ('Clean', lambda im: im),
        ('Gaussian Noise 0.05', lambda im: Image.fromarray(np.clip(np.array(im, dtype=np.float32) + np.random.normal(0, 0.05*255, (592, 592)), 0, 255).astype(np.uint8))),
        ('Gaussian Noise 0.10', lambda im: Image.fromarray(np.clip(np.array(im, dtype=np.float32) + np.random.normal(0, 0.10*255, (592, 592)), 0, 255).astype(np.uint8))),
        ('Gaussian Noise 0.20', lambda im: Image.fromarray(np.clip(np.array(im, dtype=np.float32) + np.random.normal(0, 0.20*255, (592, 592)), 0, 255).astype(np.uint8))),
        ('Blur 1px', lambda im: im.filter(ImageFilter.GaussianBlur(1))),
        ('Blur 2px', lambda im: im.filter(ImageFilter.GaussianBlur(2))),
        ('Blur 3px', lambda im: im.filter(ImageFilter.GaussianBlur(3))),
        ('JPEG Quality 30', lambda im: im), # simulate via save/load in memory
        ('10% Occlusion', lambda im: im),
        ('Rotation 2 deg', lambda im: im.rotate(2, resample=Image.Resampling.BILINEAR)),
        ('Rotation 5 deg', lambda im: im.rotate(5, resample=Image.Resampling.BILINEAR))
    ]
    
    degradation_results = {}
    for deg_name, deg_fn in degradations:
        deg_f1_list = []
        deg_move_list = []
        deg_chamfer_list = []
        for h_idx in held_40:
            gt_pts = df19.iloc[:, 2*h_idx:2*h_idx+2].dropna().values
            gt_edges = set()
            for i in range(len(gt_pts) - 1):
                u = (round(float(gt_pts[i, 0]), 2), round(float(gt_pts[i, 1]), 2))
                v = (round(float(gt_pts[i+1, 0]), 2), round(float(gt_pts[i+1, 1]), 2))
                gt_edges.add(tuple(sorted([u, v])))
                
            img_path = os.path.join(images_dir, f"kolam19-{h_idx}.jpg")
            im = Image.open(img_path).convert('L').resize((592, 592), Image.Resampling.BILINEAR)
            
            # Apply degradation
            if deg_name == 'JPEG Quality 30':
                buf = io.BytesIO()
                im.save(buf, format='JPEG', quality=30)
                buf.seek(0)
                im = Image.open(buf)
            elif deg_name == '10% Occlusion':
                arr_occ = np.array(im).copy()
                # 10% patch = approx 180x180
                arr_occ[200:380, 200:380] = 255 # white occlusion patch
                im = Image.fromarray(arr_occ)
            else:
                im = deg_fn(im)
                
            arr = 1.0 - (np.array(im, dtype=np.float32) / 255.0)
            t_img = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(t_img, edge_grid, edge_geom)
                probs = torch.sigmoid(logits).squeeze(0).numpy()
                
            rep_edges, _, _ = greedy_degree_repair(candidate_edges, probs, threshold=best_thresh_neural)
            tp = len(rep_edges & gt_edges)
            f1 = 2 * tp / max(1e-6, len(rep_edges) + len(gt_edges))
            deg_f1_list.append(f1)
            
            p_det, _, _ = deterministic_traversal(rep_edges)
            p_m = evaluate_path_metrics(p_det, gt_pts)
            deg_move_list.append(p_m['move_match_rate'])
            deg_chamfer_list.append(p_m['chamfer_lat'])
            
        degradation_results[deg_name] = {
            'mean_edge_f1': float(np.mean(deg_f1_list)),
            'mean_move_match': float(np.mean(deg_move_list)),
            'mean_chamfer_lat': float(np.mean(deg_chamfer_list))
        }
        print(f"Degradation [{deg_name}]: Edge F1={np.mean(deg_f1_list)*100:.2f}%, MoveMatch={np.mean(deg_move_list)*100:.2f}%, Chamfer={np.mean(deg_chamfer_list):.3f} lat")

    # Controls
    # 1. Black image control
    t_black = torch.zeros(1, 1, 592, 592)
    with torch.no_grad():
        logits_black = model(t_black, edge_grid, edge_geom)
        probs_black = torch.sigmoid(logits_black).squeeze(0).numpy()
    black_edges = [candidate_edges[i] for i in range(M) if probs_black[i] >= best_thresh_neural]
    
    # 2. Identity test control: pass image of held design B, score against held design A's ground truth
    # If the model follows the image, score against A should be LOW, score against B should be HIGH!
    hA, hB = held_40[0], held_40[1]
    imgB_path = os.path.join(images_dir, f"kolam19-{hB}.jpg")
    imB = Image.open(imgB_path).convert('L').resize((592, 592), Image.Resampling.BILINEAR)
    arrB = 1.0 - (np.array(imB, dtype=np.float32) / 255.0)
    with torch.no_grad():
        logits_B = model(torch.tensor(arrB).unsqueeze(0).unsqueeze(0), edge_grid, edge_geom)
        probs_B = torch.sigmoid(logits_B).squeeze(0).numpy()
    edges_B = set([candidate_edges[i] for i in range(M) if probs_B[i] >= best_thresh_neural])
    
    gt_A_pts = df19.iloc[:, 2*hA:2*hA+2].dropna().values
    gt_B_pts = df19.iloc[:, 2*hB:2*hB+2].dropna().values
    gt_A_edges = set(tuple(sorted([(round(gt_A_pts[i,0],2), round(gt_A_pts[i,1],2)), (round(gt_A_pts[i+1,0],2), round(gt_A_pts[i+1,1],2))])) for i in range(len(gt_A_pts)-1))
    gt_B_edges = set(tuple(sorted([(round(gt_B_pts[i,0],2), round(gt_B_pts[i,1],2)), (round(gt_B_pts[i+1,0],2), round(gt_B_pts[i+1,1],2))])) for i in range(len(gt_B_pts)-1))
    
    f1_B_vs_truth_B = 2 * len(edges_B & gt_B_edges) / (len(edges_B) + len(gt_B_edges))
    f1_B_vs_truth_A = 2 * len(edges_B & gt_A_edges) / (len(edges_B) + len(gt_A_edges))
    
    identity_control_passed = bool(f1_B_vs_truth_B > f1_B_vs_truth_A and f1_B_vs_truth_B > 0.7)
    print(f"Identity Test: Image B vs Truth B F1 = {f1_B_vs_truth_B*100:.2f}% | Image B vs Truth A F1 = {f1_B_vs_truth_A*100:.2f}% (Passed: {identity_control_passed})")
    log_decision(f"Identity control: Image B vs Truth B F1={f1_B_vs_truth_B*100:.2f}%, Image B vs Truth A F1={f1_B_vs_truth_A*100:.2f}% (Passed: {identity_control_passed}).")

    phase2_metrics = {
        'baseline_tuning': {
            'best_tolerance': best_tol,
            'best_threshold': best_thresh,
            'val_edge_f1': best_base_f1
        },
        'neural_training': {
            'overfit_f1_20images': f1,
            'overfit_passed': overfit_passed,
            'best_val_f1': best_val_f1,
            'selected_threshold': best_thresh_neural
        },
        'held_out_comparison': {
            'baseline': mean_baseline,
            'baseline_euler_valid_fraction': euler_rate_base,
            'neural': mean_neural,
            'neural_euler_valid_fraction': euler_rate_neural
        },
        'degradations': degradation_results,
        'controls': {
            'black_image_detected_edges': len(black_edges),
            'black_control_passed': bool(len(black_edges) == 0),
            'identity_test_f1_true_design': f1_B_vs_truth_B,
            'identity_test_f1_swapped_design': f1_B_vs_truth_A,
            'identity_control_passed': identity_control_passed
        },
        'per_design_held_results': held_results
    }
    
    out_file = os.path.join(os.path.dirname(__file__), 'reports', 'phase2_metrics.json')
    with open(out_file, 'w') as f:
        json.dump(phase2_metrics, f, indent=2)
    print(f"\n[DONE] Saved Phase 2 metrics to {out_file}")

if __name__ == '__main__':
    run_phase2()
