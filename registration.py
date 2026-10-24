"""
registration.py:
Detects pulli dots/lattice cells in Kolam images, matches to logical lattice nodes,
and solves the logical->pixel affine mapping via least squares.
Also verifies overlay agreement scores using distance-transform tolerance.
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from scipy.optimize import minimize
from skimage import measure

# Canonical affine parameters per family (fitted by least squares on distance transform)
AFFINE_PARAMS = {
    'kolam19': {
        'sx': 10.7080,
        'sy': 11.0002,
        'tx': 259.79,
        'ty': 240.60,
        'canvas_dim': 500,
        'coord_bound': 18.5
    },
    'kolam29': {
        'sx': 10.4549,
        'sy': 10.6664,
        'tx': 359.68,
        'ty': 340.21,
        'canvas_dim': 700,
        'coord_bound': 28.5
    },
    'kolam109': {
        'sx': 6.4871,
        'sy': 6.5479,
        'tx': 759.83,
        'ty': 740.19,
        'canvas_dim': 1500,
        'coord_bound': 108.5
    }
}

def detect_lattice_nodes(binary_stroke, family='kolam19'):
    """
    Detects the pulli dot / loop lattice cell centers by finding the enclosed
    regions inside stroke loops and computing their centroids.
    """
    binary_stroke = np.asarray(binary_stroke, dtype=bool)
    filled = binary_fill_holes(binary_stroke)
    holes = filled & (~binary_stroke)
    lbl = measure.label(holes)
    props = measure.regionprops(lbl)
    
    centroids = np.array([[p.centroid[1], p.centroid[0]] for p in props], dtype=np.float32)
    return centroids

def solve_affine_registration(pts_logical, img_array, family='kolam19'):
    """
    Solves least-squares affine parameters mapping logical coordinates to image pixels.
    Uses distance transform to stroke foreground.
    """
    h, w = img_array.shape
    stroke = (img_array < 200)
    dt = distance_transform_edt(~stroke)
    
    defaults = AFFINE_PARAMS[family]
    s_init_x = defaults['sx']
    s_init_y = defaults['sy']
    t_init_x = defaults['tx']
    t_init_y = defaults['ty']
    
    def cost_fn(params):
        sx, sy, tx, ty = params
        px = np.clip(np.round(sx * pts_logical[:, 0] + tx).astype(int), 0, w - 1)
        py = np.clip(np.round(sy * pts_logical[:, 1] + ty).astype(int), 0, h - 1)
        return float(np.mean(dt[py, px]**2))
        
    res = minimize(cost_fn, [s_init_x, s_init_y, t_init_x, t_init_y], method='Nelder-Mead')
    sx, sy, tx, ty = res.x
    
    px = np.clip(np.round(sx * pts_logical[:, 0] + tx).astype(int), 0, w - 1)
    py = np.clip(np.round(sy * pts_logical[:, 1] + ty).astype(int), 0, h - 1)
    res_px = float(np.mean(dt[py, px]))
    mean_scale = float((sx + sy) / 2.0)
    res_lat = res_px / mean_scale
    
    tol_px = 0.25 * mean_scale
    agreement = float(np.mean(dt[py, px] <= tol_px))
    
    return {
        'sx': float(sx),
        'sy': float(sy),
        'tx': float(tx),
        'ty': float(ty),
        'mean_scale': mean_scale,
        'residual_px': res_px,
        'residual_lat': res_lat,
        'agreement_score': agreement,
        'tol_px': tol_px
    }

def logical_to_pixel(pts_logical, family='kolam19', custom_params=None):
    """
    Transforms logical (x, y) coordinates to pixel coordinates (px, py).
    """
    p = custom_params or AFFINE_PARAMS[family]
    sx, sy = p['sx'], p['sy']
    tx, ty = p['tx'], p['ty']
    
    pts = np.asarray(pts_logical, dtype=np.float32)
    px = sx * pts[..., 0] + tx
    py = sy * pts[..., 1] + ty
    return np.column_stack([px, py]) if pts.ndim == 2 else (px, py)

def pixel_to_logical(pts_pixel, family='kolam19', custom_params=None):
    """
    Transforms pixel coordinates (px, py) to logical coordinates (x, y).
    """
    p = custom_params or AFFINE_PARAMS[family]
    sx, sy = p['sx'], p['sy']
    tx, ty = p['tx'], p['ty']
    
    pts = np.asarray(pts_pixel, dtype=np.float32)
    lx = (pts[..., 0] - tx) / sx
    ly = (pts[..., 1] - ty) / sy
    return np.column_stack([lx, ly]) if pts.ndim == 2 else (lx, ly)

def evaluate_pair1_agreement(zip_path="one-stroke-dotted-pulli-kolam.zip", images_dir="data/images"):
    """
    Evaluates Pair 1 overlay on 5 images per family using distance transform
    at 0.25-lattice-unit tolerance.
    """
    results = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        for fam in ['kolam19', 'kolam29', 'kolam109']:
            csv_path = [n for n in z.namelist() if n.endswith(f'{fam}.csv')][0]
            df = pd.read_csv(io.BytesIO(z.read(csv_path)))
            p1 = df.iloc[:, 0:2].dropna().values
            
            p = AFFINE_PARAMS[fam]
            px, py = logical_to_pixel(p1, fam)
            dim = p['canvas_dim']
            px = np.clip(np.round(px).astype(int), 0, dim - 1)
            py = np.clip(np.round(py).astype(int), 0, dim - 1)
            tol_px = 0.25 * p['sx']
            
            fam_scores = []
            fam_residuals_px = []
            fam_residuals_lat = []
            
            for idx in range(5):
                img_f = os.path.join(images_dir, f'{fam}-{idx}.jpg')
                if os.path.exists(img_f):
                    arr = np.array(Image.open(img_f).convert('L'))
                    stroke = (arr < 200)
                    dt = distance_transform_edt(~stroke)
                    dists = dt[py, px]
                    fam_scores.append(float(np.mean(dists <= tol_px)))
                    fam_residuals_px.append(float(np.mean(dists)))
                    fam_residuals_lat.append(float(np.mean(dists) / p['sx']))
                    
            results[fam] = {
                'scale_px_per_lat': (p['sx'] + p['sy']) / 2.0,
                'tol_px': tol_px,
                'mean_agreement': float(np.mean(fam_scores)),
                'mean_residual_px': float(np.mean(fam_residuals_px)),
                'mean_residual_lat': float(np.mean(fam_residuals_lat)),
                'individual_scores': fam_scores
            }
    return results

if __name__ == '__main__':
    res = evaluate_pair1_agreement()
    for fam, d in res.items():
        print(f"{fam}: Scale={d['scale_px_per_lat']:.3f} px/lat | Residual={d['mean_residual_px']:.3f} px ({d['mean_residual_lat']:.4f} lat) | Agreement={d['mean_agreement']*100:.2f}% (tol={d['tol_px']:.2f}px)")
