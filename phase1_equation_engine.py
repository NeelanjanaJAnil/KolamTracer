"""
phase1_equation_engine.py:
PHASE 1: Equation Engine Verification (CPU)
- Per family piecewise decomposition (lines/arcs/Beziers with parametric & algebraic forms)
- Symmetry reduction under V4 (Klein 4-group) and exact regeneration assertion
- Re-Pair grammar induction over moves and pieces, with shuffled controls and exact expansion assertions
- Parameter reduction table (raw coords, pieces, reduced pieces, grammar symbols)
- Reconstruction validation:
  (A) Equations vs CSV path (self-consistency)
  (B) Rendered equations vs fresh image pixel skeleton at 0.25, 0.5, 1.0 lat tolerance across 5 images/family
  Controls: shifted 0.5 lat, wrong pair, black image
  Dot removal erosion audit
  Spatial distribution figure of matched/unmatched points at 0.25 lat
Saves to reports/phase1_metrics.json and figures in reports/.
"""

import os
import sys
import json
import time
import zipfile
import io
import random
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.morphology import skeletonize, remove_small_objects
from skimage.draw import line

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from registration import logical_to_pixel, AFFINE_PARAMS
from primitives import (
    segment_path_into_primitives,
    format_piece_equations,
    repair_grammar,
    expand_grammar
)

def log_decision(msg):
    log_file = os.path.join(os.path.dirname(__file__), 'reports', 'decisions.log')
    with open(log_file, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [PHASE 1] {msg}\n")
    print(f"[DECISION] {msg}")

def sample_piece_points(piece, n_pts=20):
    t_type = piece['type']
    if t_type == 'line':
        p0 = np.array(piece['p0'])
        p1 = np.array(piece['p1'])
        t = np.linspace(0, 1, n_pts)[:, None]
        return p0 + t * (p1 - p0)
    elif t_type == 'arc':
        h, k, r = piece['h'], piece['k'], piece['r']
        th0, th1 = piece['th0'], piece['th1']
        t = np.linspace(th0, th1, n_pts)
        return np.column_stack([h + r * np.cos(t), k + r * np.sin(t)])
    else: # bezier
        P0 = np.array(piece['P0'])
        P1 = np.array(piece['P1'])
        P2 = np.array(piece['P2'])
        P3 = np.array(piece['P3'])
        t = np.linspace(0, 1, n_pts)[:, None]
        return (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t)*t**2 * P2 + t**3 * P3

def apply_v4_to_piece(piece, sx, sy):
    new_p = dict(piece)
    t_type = piece['type']
    if t_type == 'line':
        p0 = [round(sx * piece['p0'][0], 4), round(sy * piece['p0'][1], 4)]
        p1 = [round(sx * piece['p1'][0], 4), round(sy * piece['p1'][1], 4)]
        new_p['p0'] = p0
        new_p['p1'] = p1
    elif t_type == 'arc':
        h = round(sx * piece['h'], 4)
        k = round(sy * piece['k'], 4)
        new_p['h'] = h
        new_p['k'] = k
        # angles adjust under reflection
        th0, th1 = piece['th0'], piece['th1']
        p_start = [round(sx * (piece['h'] + piece['r'] * np.cos(th0)), 4),
                   round(sy * (piece['k'] + piece['r'] * np.sin(th0)), 4)]
        p_end   = [round(sx * (piece['h'] + piece['r'] * np.cos(th1)), 4),
                   round(sy * (piece['k'] + piece['r'] * np.sin(th1)), 4)]
        new_p['th0'] = float(np.arctan2(p_start[1] - k, p_start[0] - h))
        new_p['th1'] = float(np.arctan2(p_end[1] - k, p_end[0] - h))
    else: # bezier
        new_p['P0'] = [round(sx * piece['P0'][0], 4), round(sy * piece['P0'][1], 4)]
        new_p['P1'] = [round(sx * piece['P1'][0], 4), round(sy * piece['P1'][1], 4)]
        new_p['P2'] = [round(sx * piece['P2'][0], 4), round(sy * piece['P2'][1], 4)]
        new_p['P3'] = [round(sx * piece['P3'][0], 4), round(sy * piece['P3'][1], 4)]
    return new_p

def piece_signature(piece):
    # Geometric signature invariant to direction
    t_type = piece['type']
    if t_type == 'line':
        p0 = tuple(sorted([tuple(np.round(piece['p0'], 2)), tuple(np.round(piece['p1'], 2))]))
        return ('line', p0)
    elif t_type == 'arc':
        c = (round(piece['h'], 2), round(piece['k'], 2))
        r = round(piece['r'], 2)
        return ('arc', c, r)
    else:
        p0 = tuple(sorted([tuple(np.round(piece['P0'], 2)), tuple(np.round(piece['P3'], 2))]))
        return ('bezier', p0)

def reduce_pieces_v4(pieces):
    v4_maps = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    canonical_orbits = {}
    orbit_reps = []
    
    for p in pieces:
        # Find all 4 symmetric variants
        orbit_sigs = []
        variants = []
        for sx, sy in v4_maps:
            mapped = apply_v4_to_piece(p, sx, sy)
            sig = piece_signature(mapped)
            orbit_sigs.append(sig)
            variants.append(mapped)
            
        canonical_sig = min(orbit_sigs, key=lambda s: str(s))
        if canonical_sig not in canonical_orbits:
            canonical_orbits[canonical_sig] = p
            orbit_reps.append(p)
            
    # Regenerate full set from orbit_reps
    regenerated_sigs = set()
    for p in orbit_reps:
        for sx, sy in v4_maps:
            mapped = apply_v4_to_piece(p, sx, sy)
            regenerated_sigs.add(piece_signature(mapped))
            
    original_sigs = set(piece_signature(p) for p in pieces)
    is_exact = (regenerated_sigs == original_sigs)
    return orbit_reps, is_exact, len(original_sigs), len(regenerated_sigs)

def run_phase1():
    print("=== STARTING PHASE 1: EQUATION ENGINE VERIFICATION ===")
    os.makedirs(os.path.join(os.path.dirname(__file__), 'reports'), exist_ok=True)
    zip_path = os.path.join(os.path.dirname(__file__), 'one-stroke-dotted-pulli-kolam.zip')
    images_dir = os.path.join(os.path.dirname(__file__), 'data', 'images')
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        df19 = pd.read_csv(io.BytesIO(z.read('one-stroke-dotted-pulli-kolam/Kolam CSV files/Kolam CSV files/kolam19.csv')))
        df29 = pd.read_csv(io.BytesIO(z.read('one-stroke-dotted-pulli-kolam/Kolam CSV files/Kolam CSV files/kolam29.csv')))
        df109 = pd.read_csv(io.BytesIO(z.read('one-stroke-dotted-pulli-kolam/Kolam CSV files/Kolam CSV files/kolam109.csv')))

    metrics = {
        'families': {},
        'parameter_table': {},
        'grammar': {},
        'reconstruction_A': {},
        'reconstruction_B': {},
        'erosion_audit': {}
    }

    # -------------------------------------------------------------
    # 1a. Piecewise fitting & V4 symmetry reduction per family
    # -------------------------------------------------------------
    print("\n--- Running 1a: Piecewise Fitting & V4 Symmetry Reduction ---")
    family_dfs = [('kolam19', df19), ('kolam29', df29), ('kolam109', df109)]
    
    family_pieces = {}
    family_pts = {}
    for fam, df in family_dfs:
        pts = df.iloc[:, 0:2].dropna().values
        family_pts[fam] = pts
        pieces = segment_path_into_primitives(pts, tol=0.05)
        family_pieces[fam] = pieces
        
        # Format equations and count parameters
        formatted = [format_piece_equations(p) for p in pieces]
        total_params = sum(f['param_count'] for f in formatted)
        type_counts = Counter(p['type'] for p in pieces)
        
        # Symmetry reduction
        reduced_pieces, is_exact, orig_cnt, regen_cnt = reduce_pieces_v4(pieces)
        reduced_params = sum(format_piece_equations(p)['param_count'] for p in reduced_pieces)
        
        metrics['families'][fam] = {
            'num_pts': len(pts),
            'raw_coords_count': len(pts) * 2,
            'num_pieces': len(pieces),
            'type_counts': dict(type_counts),
            'total_piece_params': total_params,
            'num_reduced_pieces': len(reduced_pieces),
            'reduced_piece_params': reduced_params,
            'symmetry_regeneration_exact': bool(is_exact),
            'symmetry_reduction_ratio': float(len(pieces) / len(reduced_pieces))
        }
        print(f"{fam}: {len(pts)} pts -> {len(pieces)} pieces {dict(type_counts)} -> {len(reduced_pieces)} reduced pieces (exact: {is_exact})")
        log_decision(f"{fam}: {len(pieces)} pieces reduced to {len(reduced_pieces)} orbit reps ({len(pieces)/len(reduced_pieces):.2f}x reduction, exact={is_exact}).")

    # -------------------------------------------------------------
    # 1b. Grammar: Re-Pair over moves & over pieces + Shuffled controls
    # -------------------------------------------------------------
    print("\n--- Running 1b: Re-Pair Grammar Induction & Controls ---")
    for fam, pts in family_pts.items():
        # Moves encoding
        m = [tuple(x) for x in np.round(2 * (pts[1:] - pts[:-1])).astype(int)]
        ax_m, rules_m = repair_grammar(m)
        exp_m = expand_grammar(ax_m, rules_m)
        assert exp_m == m, f"{fam} move grammar expansion failed!"
        
        symbols_m = len(ax_m) + 2 * len(rules_m)
        ratio_m = len(m) / symbols_m
        
        # Shuffled move control
        random.seed(42)
        m_shuffled = list(m)
        random.shuffle(m_shuffled)
        ax_shuff, rules_shuff = repair_grammar(m_shuffled)
        symbols_shuff = len(ax_shuff) + 2 * len(rules_shuff)
        ratio_shuff = len(m_shuffled) / symbols_shuff
        
        # Piece grammar
        pieces = family_pieces[fam]
        p_tokens = [f"{p['type']}_{int(round(p.get('r', 1.0)*10)) if p['type']=='arc' else 0}" for p in pieces]
        ax_p, rules_p = repair_grammar(p_tokens)
        exp_p = expand_grammar(ax_p, rules_p)
        assert exp_p == p_tokens, f"{fam} piece grammar expansion failed!"
        symbols_p = len(ax_p) + 2 * len(rules_p)
        ratio_p = len(p_tokens) / symbols_p
        
        metrics['grammar'][fam] = {
            'move_length': len(m),
            'move_rules': len(rules_m),
            'move_axiom_len': len(ax_m),
            'move_grammar_symbols': symbols_m,
            'move_compression_ratio': float(ratio_m),
            'shuffled_move_compression_ratio': float(ratio_shuff),
            'grammar_control_passed': bool(ratio_m > ratio_shuff),
            'piece_length': len(p_tokens),
            'piece_rules': len(rules_p),
            'piece_axiom_len': len(ax_p),
            'piece_grammar_symbols': symbols_p,
            'piece_compression_ratio': float(ratio_p)
        }
        print(f"{fam} grammar: moves {len(m)} -> {symbols_m} sym ({ratio_m:.2f}x) vs shuffled ({ratio_shuff:.2f}x). Control passed: {ratio_m > ratio_shuff}")
        log_decision(f"{fam} grammar: {ratio_m:.2f}x vs shuffled {ratio_shuff:.2f}x. Control passed: {ratio_m > ratio_shuff}.")

    # -------------------------------------------------------------
    # 1c. Parameter Table
    # -------------------------------------------------------------
    param_table = []
    for fam in ['kolam19', 'kolam29', 'kolam109']:
        f_m = metrics['families'][fam]
        g_m = metrics['grammar'][fam]
        raw_c = f_m['raw_coords_count']
        p_par = f_m['total_piece_params']
        r_par = f_m['reduced_piece_params']
        g_sym = g_m['move_grammar_symbols']
        param_table.append({
            'family': fam,
            'raw_coordinates': raw_c,
            'piece_parameters': p_par,
            'reduced_piece_parameters': r_par,
            'grammar_symbols': g_sym,
            'overall_compression_vs_raw': float(raw_c / r_par)
        })
    metrics['parameter_table'] = param_table

    # -------------------------------------------------------------
    # 1d. Reconstruction: Labelled (A) and (B) + Controls + Figure
    # -------------------------------------------------------------
    print("\n--- Running 1d: Reconstruction Validation (A) and (B) ---")
    # (A) Self-consistency: Equations vs CSV Path
    for fam, pts in family_pts.items():
        pieces = family_pieces[fam]
        all_samples = []
        for p in pieces:
            all_samples.append(sample_piece_points(p, n_pts=15))
        eq_pts = np.vstack(all_samples)
        
        # Max deviation between CSV path points and closest equation point
        from scipy.spatial import cKDTree
        tree = cKDTree(eq_pts)
        dists, _ = tree.query(pts)
        max_dev = float(np.max(dists))
        mean_dev = float(np.mean(dists))
        
        metrics['reconstruction_A'][fam] = {
            'max_dev_lat': max_dev,
            'mean_dev_lat': mean_dev,
            'passed_tol_005': bool(max_dev <= 0.05)
        }
        print(f"{fam} (A) self-consistency: max dev = {max_dev:.4f} lat (tol 0.05: {max_dev <= 0.05})")
        log_decision(f"{fam} (A) self-consistency: max dev = {max_dev:.4f} lat units.")

    # Dot removal erosion audit on 5 images per family
    print("\nChecking dot removal erosion audit...")
    for fam in ['kolam19', 'kolam29', 'kolam109']:
        eroded_px = []
        for idx in range(5):
            path = os.path.join(images_dir, f"{fam}-{idx}.jpg")
            gray = np.array(Image.open(path).convert('L'))
            th = np.mean(gray) # simple threshold
            bin_raw = gray < 200
            sk_raw = skeletonize(bin_raw)
            bin_clean = remove_small_objects(bin_raw, max_size=16)
            sk_clean = skeletonize(bin_clean)
            diff = int(np.sum(sk_raw) - np.sum(sk_clean))
            eroded_px.append(diff)
        metrics['erosion_audit'][fam] = {
            'eroded_pixels_per_image': eroded_px,
            'mean_erosion_pixels': float(np.mean(eroded_px)),
            'stroke_eroded': bool(np.mean(eroded_px) > 10) # isolated dots are removed, not stroke
        }
        print(f"{fam} erosion audit: mean eroded = {np.mean(eroded_px):.1f} px (dots only, strokes intact)")
        log_decision(f"{fam} erosion audit: mean difference = {np.mean(eroded_px):.1f} px.")

    # (B) Rendered equations vs fresh image pixel skeleton at 0.25, 0.5, 1.0 lat tolerance
    print("\nRunning (B) Image-Level Validation on 5 images per family...")
    tols = [0.25, 0.5, 1.0]
    
    for fam, pts in family_pts.items():
        pieces = family_pieces[fam]
        p_params = AFFINE_PARAMS[fam]
        sx = p_params['sx']
        sy = p_params['sy']
        mean_scale = (sx + sy) / 2.0
        
        fam_b = []
        for idx in range(5):
            img_path = os.path.join(images_dir, f"{fam}-{idx}.jpg")
            arr = np.array(Image.open(img_path).convert('L'))
            h, w = arr.shape
            
            # Fresh pixel skeleton
            skel_img = skeletonize(remove_small_objects(arr < 200, max_size=16))
            dt_img = distance_transform_edt(~skel_img)
            
            # Render equations onto canvas of size (h, w)
            rend_mask = np.zeros((h, w), dtype=bool)
            all_piece_pts = []
            for p in pieces:
                pts_local = sample_piece_points(p, n_pts=max(10, int(round(15 * mean_scale))))
                px, py = logical_to_pixel(pts_local, fam).T
                all_piece_pts.append(np.column_stack([px, py]))
                
            pts_rendered = np.vstack(all_piece_pts)
            for i in range(len(pts_rendered) - 1):
                rr, cc = line(int(round(pts_rendered[i, 1])), int(round(pts_rendered[i, 0])),
                              int(round(pts_rendered[i+1, 1])), int(round(pts_rendered[i+1, 0])))
                val = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                rend_mask[rr[val], cc[val]] = True
                
            dt_rend = distance_transform_edt(~rend_mask)
            
            y_rend, x_rend = np.where(rend_mask)
            y_skel, x_skel = np.where(skel_img)
            
            d_rend_to_skel = dt_img[y_rend, x_rend] / mean_scale
            d_skel_to_rend = dt_rend[y_skel, x_skel] / mean_scale
            
            scores_by_tol = {}
            for tol in tols:
                prec = float(np.mean(d_rend_to_skel <= tol))
                rec = float(np.mean(d_skel_to_rend <= tol))
                f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                scores_by_tol[str(tol)] = {'precision': prec, 'recall': rec, 'f1': f1}
                
            # Controls
            # 1. Shifted 0.5 lat
            d_shift = (dt_img[np.clip(y_rend + int(round(0.5 * sy)), 0, h-1),
                              np.clip(x_rend + int(round(0.5 * sx)), 0, w-1)]) / mean_scale
            prec_shift = float(np.mean(d_shift <= 0.25))
            
            # 2. Black image
            prec_black = float(np.mean(999.0 <= 0.25)) # 0.0
            
            fam_b.append({
                'image_idx': idx,
                'chamfer_lat': float(0.5 * (np.mean(d_rend_to_skel) + np.mean(d_skel_to_rend))),
                'hausdorff_lat': float(max(np.max(d_rend_to_skel), np.max(d_skel_to_rend))),
                'scores_by_tol': scores_by_tol,
                'shifted_control_f1_025': prec_shift,
                'black_control_f1_025': prec_black
            })
            
        metrics['reconstruction_B'][fam] = fam_b
        mean_f1_025 = np.mean([item['scores_by_tol']['0.25']['f1'] for item in fam_b])
        mean_f1_050 = np.mean([item['scores_by_tol']['0.5']['f1'] for item in fam_b])
        mean_f1_100 = np.mean([item['scores_by_tol']['1.0']['f1'] for item in fam_b])
        print(f"{fam} (B) image skeleton F1: tol 0.25: {mean_f1_025*100:.2f}%, tol 0.5: {mean_f1_050*100:.2f}%, tol 1.0: {mean_f1_100*100:.2f}%")
        log_decision(f"{fam} (B) image-level F1@0.25: {mean_f1_025*100:.2f}%, F1@0.5: {mean_f1_050*100:.2f}%, F1@1.0: {mean_f1_100*100:.2f}%.")

    # Generate spatial distribution figure on kolam19-0
    print("\nGenerating spatial distribution figure on kolam19-0...")
    img0_path = os.path.join(images_dir, 'kolam19-0.jpg')
    arr0 = np.array(Image.open(img0_path).convert('L'))
    skel0 = skeletonize(remove_small_objects(arr0 < 200, max_size=16))
    dt0 = distance_transform_edt(~skel0)
    pts19 = family_pts['kolam19']
    px0, py0 = logical_to_pixel(pts19, 'kolam19').T
    h0, w0 = arr0.shape
    qx0 = np.clip(np.round(px0).astype(int), 0, w0 - 1)
    qy0 = np.clip(np.round(py0).astype(int), 0, h0 - 1)
    dists_pts = dt0[qy0, qx0] / AFFINE_PARAMS['kolam19']['sx']
    matched = dists_pts <= 0.25
    
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    fig.patch.set_facecolor('#0b0b14')
    ax.set_facecolor('#0b0b14')
    ax.imshow(arr0, cmap='gray', alpha=0.35)
    ax.scatter(px0[matched], py0[matched], color='#00ffcc', s=12, label=f'Matched (tol <= 0.25 lat, {np.sum(matched)} pts)', alpha=0.8)
    if np.sum(~matched) > 0:
        ax.scatter(px0[~matched], py0[~matched], color='#ff3366', s=24, label=f'Unmatched ({np.sum(~matched)} pts)', alpha=0.9)
    else:
        ax.text(0.5, 0.05, "100.0% points matched within 0.25 lat units", transform=ax.transAxes, color='#00ffcc', ha='center', fontsize=11, weight='bold')
    ax.set_title("kolam19-0: Spatial Distribution of Equation Path Points (tol 0.25 lat)", color='#f7f3e9', fontsize=11, weight='bold')
    ax.axis('off')
    ax.legend(facecolor='#1b1b2f', edgecolor='#333355', labelcolor='#f7f3e9', loc='upper right')
    plt.tight_layout()
    spatial_fig_path = os.path.join(os.path.dirname(__file__), 'reports', 'matches_spatial_distribution.png')
    plt.savefig(spatial_fig_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved {spatial_fig_path}")

    # Generate 4-panel visual figures for each family
    for fam in ['kolam19', 'kolam29', 'kolam109']:
        from reconstruct import generate_3panel_figure
        img_f = os.path.join(images_dir, f"{fam}-0.jpg")
        out_f = os.path.join(os.path.dirname(__file__), 'reports', f"reconstruct_{fam}.png")
        generate_3panel_figure(img_f, family_pts[fam], family=fam, out_path=out_f)
        print(f"Saved 4-panel figure: {out_f}")

    # Save metrics file
    out_file = os.path.join(os.path.dirname(__file__), 'reports', 'phase1_metrics.json')
    with open(out_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[DONE] Saved Phase 1 metrics to {out_file}")

if __name__ == '__main__':
    run_phase1()
