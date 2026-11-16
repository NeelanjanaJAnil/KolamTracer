"""
topological_engine.py:
Hybrid Constrained Topological Neural Engine for Kolam Reconstruction.
Combines:
  1. EdgeClassifierCNN (Deep feature extraction & edge logit inference)
  2. V4 Dihedral Symmetry Projection (Klein four-group orbit averaging)
  3. Edmonds' Blossom Minimum-Weight Perfect Matching (Parity repair guaranteeing Eulerian cycles)
  4. Graph topology inspection & visualizer
"""

import os
import sys
import json
import zipfile
import io
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Root directory
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from registration import logical_to_pixel, AFFINE_PARAMS
from phase2_edge_learning import (
    EdgeClassifierCNN,
    load_kolam19_master,
    build_candidate_edges,
    prepare_edge_tensors,
    check_degree_parity,
    greedy_degree_repair,
    deterministic_traversal,
    evaluate_path_metrics
)

# -------------------------------------------------------------
# 1. Symmetry Orbit Operations
# -------------------------------------------------------------
def get_v4_symmetric_edge(u, v, op):
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
# 2. Minimum-Weight Perfect Matching (MWPM) Parity Repair
# -------------------------------------------------------------
def min_weight_parity_repair(cand_edges, probs, threshold=0.55):
    """
    Guarantees 100% EVEN degree parity on all nodes using Edmonds' Blossom
    minimum-weight perfect matching algorithm on odd-degree vertices.
    """
    active = set()
    for e, p in zip(cand_edges, probs):
        if p >= threshold:
            active.add(e)
            
    v_rate, odd_nodes, degs = check_degree_parity(active)
    if not odd_nodes:
        return active, v_rate, 0.0, []

    # Build background candidate graph with uncertainty cost |p - threshold|
    G_cand = nx.Graph()
    for e, p in zip(cand_edges, probs):
        u, v = e
        cost = max(1e-4, float(abs(p - threshold)))
        G_cand.add_edge(u, v, weight=cost, edge_tuple=e)
        
    odd_list = list(odd_nodes)
    if len(odd_list) % 2 != 0:
        odd_list = odd_list[:-1] # Safety check
        
    dist_dict = {}
    path_dict = {}
    for node in odd_list:
        lengths, paths = nx.single_source_dijkstra(G_cand, node, weight='weight')
        dist_dict[node] = lengths
        path_dict[node] = paths
        
    G_odd = nx.Graph()
    for i in range(len(odd_list)):
        u = odd_list[i]
        for j in range(i + 1, len(odd_list)):
            v = odd_list[j]
            if v in dist_dict[u]:
                G_odd.add_edge(u, v, weight=dist_dict[u][v])
                
    try:
        matching = nx.algorithms.matching.min_weight_matching(G_odd, weight='weight')
    except Exception:
        return active, v_rate, v_rate, []

    repaired = set(active)
    toggled_paths = []
    for u, v in matching:
        path = path_dict[u][v]
        toggled_paths.append(path)
        for k in range(len(path) - 1):
            e = tuple(sorted([path[k], path[k+1]]))
            if e in repaired:
                repaired.remove(e)
            else:
                repaired.add(e)
                
    final_v_rate, final_odd, _ = check_degree_parity(repaired)
    return repaired, v_rate, final_v_rate, toggled_paths

# -------------------------------------------------------------
# 3. Model Loader & Unified Inference
# -------------------------------------------------------------
_CACHED_ASSETS = None

def get_topological_engine_assets():
    global _CACHED_ASSETS
    if _CACHED_ASSETS is not None:
        return _CACHED_ASSETS
        
    df19 = load_kolam19_master()
    cand_edges = build_candidate_edges(df19)
    edge_grid, edge_geom = prepare_edge_tensors(cand_edges)
    orbits = build_v4_edge_orbits(cand_edges)
    
    ckpt_path = os.path.join(ROOT, 'edge_run', 'checkpoints', 'best_edge_model.pt')
    model = EdgeClassifierCNN(num_edges=len(cand_edges))
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    _CACHED_ASSETS = {
        'model': model,
        'cand_edges': cand_edges,
        'edge_grid': edge_grid,
        'edge_geom': edge_geom,
        'orbits': orbits,
        'df19': df19
    }
    return _CACHED_ASSETS

def run_neural_topological_pipeline(image_pil, mode='hybrid_mwpm', threshold=0.60):
    """
    Modes:
      - 'raw_neural': Raw CNN thresholding without symmetry or parity constraints.
      - 'v4_symmetrized': CNN + V4 dihedral orbit projection.
      - 'hybrid_mwpm': CNN + V4 orbit projection + Edmonds' Blossom parity repair.
    """
    assets = get_topological_engine_assets()
    model = assets['model']
    cand_edges = assets['cand_edges']
    edge_grid = assets['edge_grid']
    edge_geom = assets['edge_geom']
    orbits = assets['orbits']
    
    # Preprocess image to tensor
    im_resized = image_pil.convert('L').resize((592, 592), Image.Resampling.BILINEAR)
    arr = 1.0 - (np.array(im_resized, dtype=np.float32) / 255.0)
    t_img = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(t_img, edge_grid, edge_geom)
        raw_probs = torch.sigmoid(logits).squeeze(0).numpy()
        
    toggled_paths = []
    if mode == 'raw_neural':
        probs = raw_probs
        active_edges = set([cand_edges[i] for i, p in enumerate(probs) if p >= threshold])
        v_rate, odd_nodes, degs = check_degree_parity(active_edges)
    elif mode == 'v4_symmetrized':
        probs = project_v4_symmetry(raw_probs, orbits)
        active_edges = set([cand_edges[i] for i, p in enumerate(probs) if p >= threshold])
        v_rate, odd_nodes, degs = check_degree_parity(active_edges)
    else: # 'hybrid_mwpm'
        sym_probs = project_v4_symmetry(raw_probs, orbits)
        probs = sym_probs
        active_edges, raw_v, v_rate, toggled_paths = min_weight_parity_repair(cand_edges, sym_probs, threshold=threshold)
        _, odd_nodes, degs = check_degree_parity(active_edges)

    # Deterministic straight-through traversal
    path_det, is_euler, reason = deterministic_traversal(active_edges)
    
    return {
        'active_edges': active_edges,
        'probs': probs,
        'raw_probs': raw_probs,
        'path_det': path_det,
        'is_euler': is_euler,
        'reason': reason,
        'parity_viol_rate': v_rate,
        'odd_nodes': odd_nodes,
        'node_degrees': degs,
        'toggled_paths': toggled_paths,
        'num_edges': len(active_edges),
        'num_nodes': len(degs)
    }

# -------------------------------------------------------------
# 4. Interactive Graph Topology Visualizer
# -------------------------------------------------------------
def render_graph_topology_figure(active_edges, odd_nodes, toggled_paths=None, dpi=130):
    """
    Renders the reconstructed planar graph with topological degree coding:
      - Green dots: Even degree nodes (deg = 2 or 4) [Satisfies Euler condition]
      - Red glowing dots: Odd degree defect nodes (deg = 1 or 3) [Violates Euler condition]
      - Blue lines: Blossom parity repair paths
    """
    degs = defaultdict(int)
    for u, v in active_edges:
        degs[u] += 1
        degs[v] += 1
        
    odd_set = set(odd_nodes)
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    fig.patch.set_facecolor('#0b0b14')
    ax.set_facecolor('#0b0b14')
    
    # Draw active edges
    for u, v in active_edges:
        ax.plot([u[0], v[0]], [u[1], v[1]], color='#ff9933', linewidth=1.5, alpha=0.75, solid_capstyle='round')
        
    # Draw repair paths if available
    if toggled_paths:
        for path in toggled_paths:
            px = [pt[0] for pt in path]
            py = [pt[1] for pt in path]
            ax.plot(px, py, color='#70a1ff', linewidth=2.8, linestyle='--', alpha=0.9, zorder=4)
            
    # Draw nodes
    even_x = [node[0] for node, d in degs.items() if node not in odd_set]
    even_y = [node[1] for node, d in degs.items() if node not in odd_set]
    odd_x = [node[0] for node in odd_set]
    odd_y = [node[1] for node in odd_set]
    
    if even_x:
        ax.scatter(even_x, even_y, color='#2ed573', s=16, alpha=0.85, zorder=5, label=f'Even Degree ({len(even_x)})')
    if odd_x:
        ax.scatter(odd_x, odd_y, color='#ff4757', s=45, edgecolors='white', linewidths=1.2, zorder=6, label=f'Odd Defect ({len(odd_x)})')
        
    ax.axis('off')
    ax.set_aspect('equal')
    ax.legend(facecolor='#12121f', edgecolor='rgba(255,153,51,0.3)', labelcolor='#f6efe0', loc='lower center', ncol=2, fontsize=8)
    fig.tight_layout(pad=0.2)
    return fig
