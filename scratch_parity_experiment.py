"""
scratch_parity_experiment.py:
Test Constrained Topological Decoding (V4 Symmetry + Graph Parity Matching)
on the 40 held-out kolam19 designs using the trained EdgeClassifierCNN checkpoint.
"""

import os
import sys
import json
import zipfile
import io
import time
import random
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

# Add project root to path
ROOT = r"c:\Users\lenovo\OneDrive\Desktop\SEM 2\MATERIALS\KolamTracer"
sys.path.insert(0, ROOT)
from registration import logical_to_pixel, AFFINE_PARAMS
from phase2_edge_learning import (
    EdgeClassifierCNN,
    load_kolam19_master,
    build_candidate_edges,
    get_splits,
    prepare_edge_tensors,
    check_degree_parity,
    greedy_degree_repair,
    deterministic_traversal,
    evaluate_path_metrics
)

# -------------------------------------------------------------
# 1. V4 Symmetry Projection
# -------------------------------------------------------------
def get_v4_symmetric_edge(u, v, op):
    # op: 'id', 'refl_x', 'refl_y', 'rot180'
    x1, y1 = u
    x2, y2 = v
    if op == 'id':
        u_p, v_p = (x1, y1), (x2, y2)
    elif op == 'refl_x':
        u_p, v_p = (-x1, y1), (-x2, y2)
    elif op == 'refl_y':
        u_p, v_p = (x1, -y1), (x2, -y2)
    elif op == 'rot180':
        u_p, v_p = (-x1, -y1), (-x2, -y2)
    return tuple(sorted([(round(u_p[0], 2), round(u_p[1], 2)),
                         (round(v_p[0], 2), round(v_p[1], 2))]))

def build_v4_edge_orbits(candidate_edges):
    cand_map = {e: i for i, e in enumerate(candidate_edges)}
    orbits = []
    visited = set()
    
    for i, (u, v) in enumerate(candidate_edges):
        e = (u, v)
        if e in visited:
            continue
        orbit = []
        for op in ['id', 'refl_x', 'refl_y', 'rot180']:
            e_sym = get_v4_symmetric_edge(u, v, op)
            if e_sym in cand_map:
                idx = cand_map[e_sym]
                if idx not in orbit:
                    orbit.append(idx)
                    visited.add(e_sym)
        orbits.append(orbit)
        
    return orbits

def project_v4_symmetry(probs, orbits):
    sym_probs = probs.copy()
    for orbit in orbits:
        mean_p = float(np.mean(probs[orbit]))
        for idx in orbit:
            sym_probs[idx] = mean_p
    return sym_probs

# -------------------------------------------------------------
# 2. Min-Weight Matching Parity Repair
# -------------------------------------------------------------
def min_weight_parity_repair(cand_edges, probs, threshold=0.5):
    """
    Finds odd-degree nodes and pairs them up via shortest path in candidate lattice
    weighted by edge uncertainty |prob - threshold|, guaranteeing 100% EVEN degree parity.
    """
    # Active edges at threshold
    active = set()
    for e, p in zip(cand_edges, probs):
        if p >= threshold:
            active.add(e)
            
    v_rate, odd_nodes, degs = check_degree_parity(active)
    if not odd_nodes:
        return active, v_rate, 0.0

    # Build background candidate graph with costs
    G_cand = nx.Graph()
    for e, p in zip(cand_edges, probs):
        u, v = e
        # If edge is currently active, turning it OFF costs (p - threshold)
        # If edge is currently inactive, turning it ON costs (threshold - p)
        # Total cost is always positive |p - threshold|
        cost = max(1e-4, float(abs(p - threshold)))
        G_cand.add_edge(u, v, weight=cost, edge_tuple=e)
        
    # We need to pair all odd nodes with each other
    # Compute all-pairs shortest paths between odd nodes in G_cand
    odd_list = list(odd_nodes)
    if len(odd_list) % 2 != 0:
        # Should not happen by Handshaking Lemma, but safety check:
        odd_list = odd_list[:-1]
        
    dist_dict = {}
    path_dict = {}
    for node in odd_list:
        lengths, paths = nx.single_source_dijkstra(G_cand, node, weight='weight')
        dist_dict[node] = lengths
        path_dict[node] = paths
        
    # Build complete distance graph on odd nodes
    G_odd = nx.Graph()
    for i in range(len(odd_list)):
        u = odd_list[i]
        for j in range(i + 1, len(odd_list)):
            v = odd_list[j]
            if v in dist_dict[u]:
                G_odd.add_edge(u, v, weight=dist_dict[u][v])
                
    # Minimum-Weight Perfect Matching (Blossom Algorithm)
    try:
        matching = nx.algorithms.matching.min_weight_matching(G_odd, weight='weight')
    except Exception as exc:
        print("Matching fallback:", exc)
        return active, v_rate, v_rate

    # Toggle edges along the shortest paths connecting matched pairs
    repaired = set(active)
    for u, v in matching:
        path = path_dict[u][v]
        for k in range(len(path) - 1):
            e = tuple(sorted([path[k], path[k+1]]))
            if e in repaired:
                repaired.remove(e)
            else:
                repaired.add(e)
                
    final_v_rate, final_odd, _ = check_degree_parity(repaired)
    return repaired, v_rate, final_v_rate

# -------------------------------------------------------------
# 3. Main Test Runner
# -------------------------------------------------------------
def run_experiment():
    print("=== Testing Topological Parity Decoding on 40 Held-out Designs ===")
    df19 = load_kolam19_master()
    cand_edges = build_candidate_edges(df19)
    train_324, val_36, held_40 = get_splits()
    edge_grid, edge_geom = prepare_edge_tensors(cand_edges)
    orbits = build_v4_edge_orbits(cand_edges)
    
    # Load model
    ckpt_path = os.path.join(ROOT, 'edge_run', 'checkpoints', 'best_edge_model.pt')
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = EdgeClassifierCNN(num_edges=len(cand_edges))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    images_dir = os.path.join(ROOT, 'data', 'images')
    
    methods = ['raw_greedy', 'v4_greedy', 'v4_mwpm_matching']
    stats = {m: {'edge_f1': [], 'parity_viol': [], 'move_match': [], 'prefix': [], 'is_euler': []} for m in methods}
    
    print(f"Running inference on {len(held_40)} held-out designs...")
    
    for count, h_idx in enumerate(held_40):
        # 1. Ground truth
        gt_pts = df19.iloc[:, 2*h_idx:2*h_idx+2].dropna().values
        gt_edges = set()
        for i in range(len(gt_pts) - 1):
            u = (round(float(gt_pts[i, 0]), 2), round(float(gt_pts[i, 1]), 2))
            v = (round(float(gt_pts[i+1, 0]), 2), round(float(gt_pts[i+1, 1]), 2))
            gt_edges.add(tuple(sorted([u, v])))
            
        # 2. Image inference
        img_path = os.path.join(images_dir, f"kolam19-{h_idx}.jpg")
        im = Image.open(img_path).convert('L').resize((592, 592), Image.Resampling.BILINEAR)
        arr = 1.0 - (np.array(im, dtype=np.float32) / 255.0)
        t_img = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(t_img, edge_grid, edge_geom)
            probs = torch.sigmoid(logits).squeeze(0).numpy()
            
        # Symmetrized probs
        sym_probs = project_v4_symmetry(probs, orbits)
        
        # Test 1: Raw Greedy (current method, threshold 0.60)
        edges_m0, _, v_m0 = greedy_degree_repair(cand_edges, probs, threshold=0.60)
        p0, euler0, _ = deterministic_traversal(edges_m0)
        m0_metrics = evaluate_path_metrics(p0, gt_pts)
        f1_0 = 2 * len(edges_m0 & gt_edges) / max(1e-6, len(edges_m0) + len(gt_edges))
        stats['raw_greedy']['edge_f1'].append(f1_0)
        stats['raw_greedy']['parity_viol'].append(v_m0)
        stats['raw_greedy']['move_match'].append(m0_metrics['move_match_rate'])
        stats['raw_greedy']['prefix'].append(m0_metrics['correct_prefix_len'])
        stats['raw_greedy']['is_euler'].append(float(euler0))
        
        # Test 2: V4 Symmetrized + Greedy
        edges_m1, _, v_m1 = greedy_degree_repair(cand_edges, sym_probs, threshold=0.60)
        p1, euler1, _ = deterministic_traversal(edges_m1)
        m1_metrics = evaluate_path_metrics(p1, gt_pts)
        f1_1 = 2 * len(edges_m1 & gt_edges) / max(1e-6, len(edges_m1) + len(gt_edges))
        stats['v4_greedy']['edge_f1'].append(f1_1)
        stats['v4_greedy']['parity_viol'].append(v_m1)
        stats['v4_greedy']['move_match'].append(m1_metrics['move_match_rate'])
        stats['v4_greedy']['prefix'].append(m1_metrics['correct_prefix_len'])
        stats['v4_greedy']['is_euler'].append(float(euler1))
        
        # Test 3: V4 Symmetrized + Min-Weight Matching Parity Repair
        edges_m2, _, v_m2 = min_weight_parity_repair(cand_edges, sym_probs, threshold=0.60)
        p2, euler2, _ = deterministic_traversal(edges_m2)
        m2_metrics = evaluate_path_metrics(p2, gt_pts)
        f1_2 = 2 * len(edges_m2 & gt_edges) / max(1e-6, len(edges_m2) + len(gt_edges))
        stats['v4_mwpm_matching']['edge_f1'].append(f1_2)
        stats['v4_mwpm_matching']['parity_viol'].append(v_m2)
        stats['v4_mwpm_matching']['move_match'].append(m2_metrics['move_match_rate'])
        stats['v4_mwpm_matching']['prefix'].append(m2_metrics['correct_prefix_len'])
        stats['v4_mwpm_matching']['is_euler'].append(float(euler2))
        
        if (count + 1) % 10 == 0:
            print(f"  Processed {count + 1}/40 designs...")
            
    print("\n" + "="*70)
    print("COMPARATIVE RESULTS ON 40 HELD-OUT DESIGNS:")
    print("="*70)
    for m in methods:
        f1 = np.mean(stats[m]['edge_f1']) * 100
        pv = np.mean(stats[m]['parity_viol']) * 100
        mm = np.mean(stats[m]['move_match']) * 100
        pr = np.mean(stats[m]['prefix'])
        eu = np.mean(stats[m]['is_euler']) * 100
        print(f"Method: {m:<20} | Edge F1: {f1:5.2f}% | Odd Parity: {pv:5.2f}% | Move Match: {mm:5.2f}% | Prefix: {pr:5.1f}/724 | Euler: {eu:4.1f}%")

if __name__ == '__main__':
    run_experiment()
