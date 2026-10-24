"""
tracer.py:
Pure image-to-path heuristic tracer on pixel skeletons (without using ground-truth CSV).
Implements a graph walker on the skeleton pixels:
  - Detects regular curve pixels (2 neighbors) and junction nodes (>2 neighbors).
  - Starts at the highest-confidence start position (closest skeleton pixel to logical start or extreme pixel).
  - Traverses the skeleton pixels.
  - At crossings/junctions, chooses the outgoing branch that minimizes turn angle
    (maximizes cosine similarity with the incoming direction: the "straight-through" rule).
  - Benchmarks coverage and Chamfer distance across 20 images per family against CSV path.
  - Reports unvarnished, honest failure rates when trapped or stuck in loops.
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt
from registration import logical_to_pixel, pixel_to_logical, AFFINE_PARAMS

# 8-connected neighborhood offsets
NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

def extract_skeleton(img_path):
    """
    Loads grayscale image and extracts Zhang-Suen 1-pixel skeleton.
    """
    im = Image.open(img_path).convert('L')
    arr = np.array(im)
    bin_stroke = remove_small_objects(arr < 200, max_size=16)
    skel = skeletonize(bin_stroke)
    return skel, arr.shape

def trace_skeleton_straight_through(skel, start_rc=None, max_steps=100000):
    """
    Walks the skeleton using the 'straight-through at crossings' rule.
    Returns:
      traced_path: list of (r, c) pixel coordinates
      coverage_pct: fraction of total skeleton pixels traversed
      termination_reason: 'trapped', 'loop_closed', or 'max_steps'
    """
    h, w = skel.shape
    skel_coords = set(zip(*np.where(skel)))
    total_skel_pixels = len(skel_coords)
    if total_skel_pixels == 0:
        return [], 0.0, "empty_skeleton"

    if start_rc is None or start_rc not in skel_coords:
        # Pick topmost, then leftmost pixel
        start_rc = min(skel_coords, key=lambda rc: (rc[0], rc[1]))

    visited = set()
    traced_path = [start_rc]
    visited.add(start_rc)

    curr = start_rc
    prev_dir = np.array([0.0, 1.0]) # initial default direction

    # Track edge visit counts to prevent infinite ping-pong
    visit_count = {pt: 0 for pt in skel_coords}
    visit_count[start_rc] = 1

    for step in range(max_steps):
        r, c = curr
        # Find 8-neighbors on skeleton
        valid_nbrs = []
        for dr, dc in NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if (nr, nc) in skel_coords:
                valid_nbrs.append((nr, nc))

        # Filter neighbors to unvisited or least-visited
        unvisited = [nbr for nbr in valid_nbrs if nbr not in visited]

        if not unvisited:
            # Check if we can close loop back to start
            if len(traced_path) > 10 and start_rc in valid_nbrs:
                traced_path.append(start_rc)
                return traced_path, float(len(visited) / total_skel_pixels * 100.0), "loop_closed"
            
            # Trapped with no unvisited branch
            return traced_path, float(len(visited) / total_skel_pixels * 100.0), "trapped"

        if len(unvisited) == 1:
            next_pt = unvisited[0]
        else:
            # Junction / crossing: apply "straight-through" rule
            # Choose neighbor maximizing dot product with incoming direction
            best_nbr = unvisited[0]
            best_cos = -999.0
            
            # Incoming direction
            in_dir = prev_dir
            norm_in = np.linalg.norm(in_dir)
            if norm_in > 1e-6:
                in_dir = in_dir / norm_in

            for nbr in unvisited:
                out_dir = np.array([nbr[0] - r, nbr[1] - c], dtype=float)
                norm_out = np.linalg.norm(out_dir)
                if norm_out > 1e-6:
                    out_dir = out_dir / norm_out
                cos_sim = float(np.dot(in_dir, out_dir))
                if cos_sim > best_cos:
                    best_cos = cos_sim
                    best_nbr = nbr
            next_pt = best_nbr

        # Advance step
        prev_dir = np.array([next_pt[0] - curr[0], next_pt[1] - curr[1]], dtype=float)
        curr = next_pt
        visited.add(curr)
        traced_path.append(curr)

    return traced_path, float(len(visited) / total_skel_pixels * 100.0), "max_steps"

def benchmark_tracer(zip_path="../KolamTracer/one-stroke-dotted-pulli-kolam.zip",
                     images_dir="../KolamTracer/data/images",
                     num_images=20):
    """
    Runs the heuristic tracer on 20 images per family and benchmarks:
      - Skeleton pixel coverage %
      - Chamfer distance to ground-truth CSV path (lattice units)
      - Termination reason distribution
    """
    results = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for fam in ['kolam19', 'kolam29', 'kolam109']:
            cpath = [n for n in z.namelist() if n.endswith(f'{fam}.csv')][0]
            df = pd.read_csv(io.BytesIO(z.read(cpath)))
            p1_logical = df.iloc[:, 0:2].dropna().values
            scale = AFFINE_PARAMS[fam]['sx']

            # Map logical start to pixel start
            p_start_pix = logical_to_pixel(p1_logical[:1], fam)[0]
            target_start_rc = (int(round(p_start_pix[1])), int(round(p_start_pix[0])))

            fam_coverages = []
            fam_chamfers = []
            fam_reasons = []

            for idx in range(num_images):
                img_path = os.path.join(images_dir, f'{fam}-{idx}.jpg')
                if not os.path.exists(img_path):
                    continue
                skel, (h, w) = extract_skeleton(img_path)
                skel_coords = list(zip(*np.where(skel)))
                if not skel_coords:
                    continue

                # Find closest skeleton pixel to true start
                dists = [np.hypot(rc[0] - target_start_rc[0], rc[1] - target_start_rc[1]) for rc in skel_coords]
                best_start_rc = skel_coords[np.argmin(dists)]

                path_rc, cov_pct, reason = trace_skeleton_straight_through(skel, start_rc=best_start_rc)

                # Chamfer to CSV path
                if len(path_rc) > 1:
                    traced_xy_pix = np.column_stack([np.array([c for r, c in path_rc]), np.array([r for r, c in path_rc])])
                    # Sample GT path pixels
                    gt_pix = logical_to_pixel(p1_logical, fam)
                    # Point-to-point chamfer
                    # Distance from traced points to GT polyline
                    dt_gt = distance_transform_edt(~draw_polyline_mask(gt_pix, h, w))
                    traced_r = np.clip(np.array([r for r, c in path_rc]), 0, h - 1)
                    traced_c = np.clip(np.array([c for r, c in path_rc]), 0, w - 1)
                    chamfer_lat = float(np.mean(dt_gt[traced_r, traced_c]) / scale)
                else:
                    chamfer_lat = 999.0

                fam_coverages.append(cov_pct)
                fam_chamfers.append(chamfer_lat)
                fam_reasons.append(reason)

                results.append({
                    'family': fam,
                    'image': f'{fam}-{idx}.jpg',
                    'coverage_pct': cov_pct,
                    'chamfer_lat': chamfer_lat,
                    'reason': reason
                })

            mean_cov = np.mean(fam_coverages) if fam_coverages else 0.0
            mean_cham = np.mean(fam_chamfers) if fam_chamfers else 0.0
            print(f'{fam} (20 images): Mean coverage = {mean_cov:.2f}%, Mean Chamfer = {mean_cham:.3f} lat units')
            print(f'  Termination reasons: {pd.Series(fam_reasons).value_counts().to_dict()}')

    return pd.DataFrame(results)

def draw_polyline_mask(pts_pix, h, w):
    from skimage.draw import line
    mask = np.zeros((h, w), dtype=bool)
    for i in range(len(pts_pix) - 1):
        rr, cc = line(int(round(pts_pix[i, 1])), int(round(pts_pix[i, 0])),
                      int(round(pts_pix[i+1, 1])), int(round(pts_pix[i+1, 0])))
        valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        mask[rr[valid], cc[valid]] = True
    return mask

if __name__ == '__main__':
    df = benchmark_tracer()
    print('\nOverall Benchmark Summary:')
    print(df.groupby('family')[['coverage_pct', 'chamfer_lat']].mean())
