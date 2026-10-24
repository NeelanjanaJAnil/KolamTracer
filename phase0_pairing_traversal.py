"""
phase0_pairing_traversal.py:
PHASE 0: Pairing + Traversal Facts
- Pair retrieval across families using fresh pixel skeletons vs ALL pairs
- Synthetic positive control & negative controls
- Deterministic traversal check on ALL true edge sets
- Ambiguity rate check on candidate edges (< 0.1 lat)
- V4 invariance check per family with asymmetric negative control
Saves output to reports/phase0_metrics.json and logs to reports/decisions.log.
"""

import os
import sys
import json
import time
import zipfile
import io
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from registration import logical_to_pixel, AFFINE_PARAMS

def log_decision(msg):
    log_file = os.path.join(os.path.dirname(__file__), 'reports', 'decisions.log')
    with open(log_file, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [PHASE 0] {msg}\n")
    print(f"[DECISION] {msg}")

def get_fresh_skeleton(img_path):
    im = Image.open(img_path).convert('L')
    arr = np.array(im)
    bin_stroke = remove_small_objects(arr < 200, max_size=16)
    skel = skeletonize(bin_stroke)
    return skel, arr.shape

def run_phase0():
    print("=== STARTING PHASE 0: PAIRING & TRAVERSAL FACTS ===")
    os.makedirs(os.path.join(os.path.dirname(__file__), 'reports'), exist_ok=True)
    
    zip_path = os.path.join(os.path.dirname(__file__), 'one-stroke-dotted-pulli-kolam.zip')
    images_dir = os.path.join(os.path.dirname(__file__), 'data', 'images')
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        df19 = pd.read_csv(io.BytesIO(z.read('one-stroke-dotted-pulli-kolam/Kolam CSV files/Kolam CSV files/kolam19.csv')))
        df29 = pd.read_csv(io.BytesIO(z.read('one-stroke-dotted-pulli-kolam/Kolam CSV files/Kolam CSV files/kolam29.csv')))
        df109 = pd.read_csv(io.BytesIO(z.read('one-stroke-dotted-pulli-kolam/Kolam CSV files/Kolam CSV files/kolam109.csv')))

    metrics = {
        'pair_retrieval': {},
        'traversal_reproduction': {},
        'ambiguity_rate': {},
        'v4_invariance': {}
    }

    # -------------------------------------------------------------
    # 0a. Pair Retrieval
    # -------------------------------------------------------------
    print("\n--- Running 0a: Pair Retrieval on fresh pixel skeletons ---")
    retrieval_tasks = [
        ('kolam19', df19, 400, 20),
        ('kolam29', df29, 100, 10),
        ('kolam109', df109, 100, 5)
    ]
    
    for fam, df, total_pairs, num_images in retrieval_tasks:
        fam_res = []
        tol_px = 0.25 * AFFINE_PARAMS[fam]['sx']
        
        # Precompute pixel coordinates for all pairs in this family
        all_pair_coords = []
        for p_idx in range(total_pairs):
            pts = df.iloc[:, 2*p_idx:2*p_idx+2].dropna().values
            px, py = logical_to_pixel(pts, fam).T
            all_pair_coords.append((px, py))
            
        print(f"Testing {fam}: {num_images} images against {total_pairs} pairs...")
        for img_idx in range(num_images):
            img_path = os.path.join(images_dir, f"{fam}-{img_idx}.jpg")
            skel, shape = get_fresh_skeleton(img_path)
            dt_skel = distance_transform_edt(~skel)
            h, w = shape
            
            scores = []
            for p_idx, (px, py) in enumerate(all_pair_coords):
                valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
                qx = np.clip(np.round(px).astype(int), 0, w - 1)
                qy = np.clip(np.round(py).astype(int), 0, h - 1)
                dists = dt_skel[qy, qx]
                sc = float(np.mean(dists <= tol_px))
                scores.append((sc, p_idx + 1))
                
            scores.sort(key=lambda x: x[0], reverse=True)
            expected_pair = img_idx + 1
            
            # Find rank of expected pair
            ranks = [idx + 1 for idx, item in enumerate(scores) if item[1] == expected_pair]
            exp_rank = ranks[0] if ranks else -1
            best_sc, best_p = scores[0]
            runner_sc, runner_p = scores[1] if len(scores) > 1 else (0.0, -1)
            margin = best_sc - runner_sc
            
            fam_res.append({
                'image_idx': img_idx,
                'expected_pair': expected_pair,
                'expected_rank': exp_rank,
                'best_pair': best_p,
                'best_score': best_sc,
                'runner_up_pair': runner_p,
                'runner_up_score': runner_sc,
                'margin': margin
            })
            
        metrics['pair_retrieval'][fam] = fam_res
        perfect = sum(1 for r in fam_res if r['expected_rank'] == 1 and r['best_score'] >= 0.99)
        print(f"{fam} retrieval: {perfect}/{num_images} retrieved as Rank 1 (best score >= 0.99)")
        log_decision(f"{fam} pair retrieval evaluated on {num_images} images: {perfect}/{num_images} Rank 1.")

    # Synthetic Positive Control: rasterize pair 1 on clean canvas, score against all 400 pairs
    print("Running Synthetic Positive Control (kolam19 pair 1 rasterized)...")
    ctrl_pts = df19.iloc[:, 0:2].dropna().values
    ctrl_px, ctrl_py = logical_to_pixel(ctrl_pts, 'kolam19').T
    h, w = 500, 500
    ctrl_skel = np.zeros((h, w), dtype=bool)
    from skimage.draw import line
    for i in range(len(ctrl_px) - 1):
        rr, cc = line(int(round(ctrl_py[i])), int(round(ctrl_px[i])),
                      int(round(ctrl_py[i+1])), int(round(ctrl_px[i+1])))
        val = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        ctrl_skel[rr[val], cc[val]] = True
    dt_ctrl = distance_transform_edt(~ctrl_skel)
    tol_px = 0.25 * AFFINE_PARAMS['kolam19']['sx']
    
    synth_scores = []
    for p_idx in range(400):
        pts = df19.iloc[:, 2*p_idx:2*p_idx+2].dropna().values
        px, py = logical_to_pixel(pts, 'kolam19').T
        qx = np.clip(np.round(px).astype(int), 0, w - 1)
        qy = np.clip(np.round(py).astype(int), 0, h - 1)
        sc = float(np.mean(dt_ctrl[qy, qx] <= tol_px))
        synth_scores.append((sc, p_idx + 1))
    synth_scores.sort(key=lambda x: x[0], reverse=True)
    
    metrics['pair_retrieval']['synthetic_positive_control'] = {
        'best_pair': synth_scores[0][1],
        'best_score': synth_scores[0][0],
        'runner_up_pair': synth_scores[1][1],
        'runner_up_score': synth_scores[1][0],
        'margin': synth_scores[0][0] - synth_scores[1][0],
        'passed': bool(synth_scores[0][1] == 1 and synth_scores[0][0] == 1.0)
    }
    print(f"Synthetic positive control passed: {metrics['pair_retrieval']['synthetic_positive_control']['passed']}")
    log_decision(f"Synthetic positive control passed: {metrics['pair_retrieval']['synthetic_positive_control']['passed']}")

    # Negative control: blank image scored against all pairs
    dt_black = np.ones((h, w), dtype=np.float32) * 999.0
    black_scores = []
    for p_idx in range(10):
        pts = df19.iloc[:, 2*p_idx:2*p_idx+2].dropna().values
        px, py = logical_to_pixel(pts, 'kolam19').T
        qx = np.clip(np.round(px).astype(int), 0, w - 1)
        qy = np.clip(np.round(py).astype(int), 0, h - 1)
        sc = float(np.mean(dt_black[qy, qx] <= tol_px))
        black_scores.append(sc)
    metrics['pair_retrieval']['black_image_control'] = {
        'mean_score': float(np.mean(black_scores)),
        'passed': bool(np.mean(black_scores) == 0.0)
    }
    log_decision(f"Black image control passed: {metrics['pair_retrieval']['black_image_control']['passed']}")

    # -------------------------------------------------------------
    # 0b. Deterministic Traversal
    # -------------------------------------------------------------
    print("\n--- Running 0b: Deterministic Traversal on ALL true edge sets ---")
    def run_deterministic_traversal(pts):
        N = len(pts)
        adj = defaultdict(list)
        edge_counts = defaultdict(int)
        for i in range(N - 1):
            u = (round(float(pts[i, 0]), 2), round(float(pts[i, 1]), 2))
            v = (round(float(pts[i+1, 0]), 2), round(float(pts[i+1, 1]), 2))
            e = tuple(sorted([u, v]))
            edge_counts[e] += 1
            adj[u].append(v)
            adj[v].append(u)
            
        start_node = (round(float(pts[0, 0]), 2), round(float(pts[0, 1]), 2))
        p1 = (round(float(pts[1, 0]), 2), round(float(pts[1, 1]), 2))
        d_in = np.array([p1[0] - start_node[0], p1[1] - start_node[1]])
        
        curr = start_node
        path = [curr]
        remaining = dict(edge_counts)
        
        first_e = tuple(sorted([start_node, p1]))
        if remaining[first_e] > 0:
            remaining[first_e] -= 1
            curr = p1
            path.append(curr)
        else:
            return False, len(path), 'First edge not available'
            
        while True:
            candidates = []
            for nxt in adj[curr]:
                e = tuple(sorted([curr, nxt]))
                if remaining[e] > 0:
                    candidates.append(nxt)
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
            
        if len(path) != N:
            return False, len(path), f'Length mismatch: {len(path)} vs {N}'
        path_arr = np.array(path)
        if not np.allclose(path_arr, pts, atol=1e-3):
            diffs = np.where(~np.isclose(path_arr, pts, atol=1e-3))[0]
            return False, len(path), f'Coords differ at step {diffs[0]}'
        return True, len(path), 'Exact match'

    traversal_tasks = [
        ('kolam19', df19, 400),
        ('kolam29', df29, 100),
        ('kolam109', df109, 100)
    ]
    for fam, df, n_pairs in traversal_tasks:
        print(f"Checking traversal on {fam} ({n_pairs} designs)...")
        exact_matches = 0
        failures = []
        for p in range(n_pairs):
            pts = df.iloc[:, 2*p:2*p+2].dropna().values
            ok, l, msg = run_deterministic_traversal(pts)
            if ok:
                exact_matches += 1
            else:
                failures.append({'pair': p + 1, 'reason': msg, 'recovered_len': l, 'expected_len': len(pts)})
        metrics['traversal_reproduction'][fam] = {
            'total_tested': n_pairs,
            'exact_matches': exact_matches,
            'exact_reproduction_rate': float(exact_matches / n_pairs),
            'failures': failures
        }
        print(f"{fam} deterministic traversal: {exact_matches}/{n_pairs} ({exact_matches/n_pairs*100:.2f}%)")
        log_decision(f"{fam} deterministic traversal: {exact_matches}/{n_pairs} exact reproductions.")

    # -------------------------------------------------------------
    # 0c. Ambiguity Rate (< 0.1 lat)
    # -------------------------------------------------------------
    print("\n--- Running 0c: Ambiguity Rate on kolam19 candidate edges ---")
    candidate_edges = set()
    for p in range(400):
        pts = df19.iloc[:, 2*p:2*p+2].dropna().values
        for i in range(len(pts) - 1):
            u = (round(float(pts[i, 0]), 2), round(float(pts[i, 1]), 2))
            v = (round(float(pts[i+1, 0]), 2), round(float(pts[i+1, 1]), 2))
            candidate_edges.add(tuple(sorted([u, v])))

    cand_list = sorted(list(candidate_edges))
    M = len(cand_list)
    cand_a = np.array([[e[0][0], e[0][1]] for e in cand_list], dtype=np.float32)
    cand_b = np.array([[e[1][0], e[1][1]] for e in cand_list], dtype=np.float32)
    mids = (cand_a + cand_b) / 2.0

    ab = cand_b - cand_a
    ab_len2 = np.sum(ab**2, axis=-1, keepdims=True)

    p_minus_a = mids[:, None, :] - cand_a[None, :, :]
    ab_bcast = ab[None, :, :]
    dot = np.sum(p_minus_a * ab_bcast, axis=-1)
    t = np.clip(dot / (ab_len2.T + 1e-9), 0.0, 1.0)
    proj = cand_a[None, :, :] + t[:, :, None] * ab_bcast
    dists = np.linalg.norm(mids[:, None, :] - proj, axis=-1)
    np.fill_diagonal(dists, 999.0)

    min_dists = np.min(dists, axis=1)
    ambiguous_count = int(np.sum(min_dists < 0.1))
    ambiguity_rate = float(ambiguous_count / M)
    min_dist_overall = float(np.min(min_dists))

    metrics['ambiguity_rate'] = {
        'total_candidate_edges': M,
        'ambiguous_count': ambiguous_count,
        'ambiguity_rate': ambiguity_rate,
        'min_distance_lattice_units': min_dist_overall,
        'passed': bool(ambiguity_rate == 0.0)
    }
    print(f"kolam19 candidate edge ambiguity rate (< 0.1 lat): {ambiguous_count}/{M} ({ambiguity_rate*100:.2f}%)")
    print(f"Minimum distance between any midpoint and other edge segment: {min_dist_overall:.4f} lat units")
    log_decision(f"Candidate edge ambiguity rate: {ambiguous_count}/{M} ({ambiguity_rate:.4f}), min separation: {min_dist_overall:.4f} lat units.")

    # -------------------------------------------------------------
    # 0d. V4 Invariance per family
    # -------------------------------------------------------------
    print("\n--- Running 0d: V4 Invariance per family ---")
    v4_maps = [
        ('Id', 1, 1),
        ('FlipX', -1, 1),
        ('FlipY', 1, -1),
        ('Rot180', -1, -1)
    ]
    def apply_v4(E, sx, sy):
        res = set()
        for u, v in E:
            u_t = (round(sx * u[0], 2), round(sy * u[1], 2))
            v_t = (round(sx * v[0], 2), round(sy * v[1], 2))
            res.add(tuple(sorted([u_t, v_t])))
        return res

    for fam, df in [('kolam19', df19), ('kolam29', df29), ('kolam109', df109)]:
        pts = df.iloc[:, 0:2].dropna().values
        edges = set()
        for i in range(len(pts) - 1):
            u = (round(float(pts[i, 0]), 2), round(float(pts[i, 1]), 2))
            v = (round(float(pts[i+1, 0]), 2), round(float(pts[i+1, 1]), 2))
            edges.add(tuple(sorted([u, v])))
            
        invariant = all(apply_v4(edges, sx, sy) == edges for _, sx, sy in v4_maps)
        
        # Negative control: drop 1 edge
        asym_edges = set(list(edges)[1:])
        asym_passed = not all(apply_v4(asym_edges, sx, sy) == asym_edges for _, sx, sy in v4_maps)
        
        metrics['v4_invariance'][fam] = {
            'num_edges': len(edges),
            'v4_invariant': bool(invariant),
            'negative_control_passed': bool(asym_passed)
        }
        print(f"{fam}: |E|={len(edges)}, V4 invariant={invariant}, Asymmetric negative control passed={asym_passed}")
        log_decision(f"{fam} V4 invariance: {invariant}, Negative control passed: {asym_passed}.")

    # Save metrics file
    out_file = os.path.join(os.path.dirname(__file__), 'reports', 'phase0_metrics.json')
    with open(out_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[DONE] Saved Phase 0 metrics to {out_file}")

if __name__ == '__main__':
    run_phase0()
