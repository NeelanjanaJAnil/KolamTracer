"""
reconstruct.py:
Renders the induced equations onto a high-resolution canvas (>= 16 px per lattice unit)
with stroke width matching the original image and glowing dots at lattice nodes.
Computes rigorous reconstruction metrics:
  - Stroke-skeleton Chamfer and Hausdorff in lattice units
  - F1 score at 0.25 and 0.5 lattice-unit tolerance
  - Path-to-original Chamfer
Saves 3-panel figures: Original | Rendered from equations | Overlay (Original cyan, Rendered magenta) + Difference map.
Also generates a max/mean deviation table across 5 images per family.
"""

import os
import io
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.draw import line
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt, binary_dilation

from registration import logical_to_pixel, AFFINE_PARAMS, detect_lattice_nodes

def render_equation_canvas(pieces, family='kolam19', scale_px_lat=16.0, stroke_px=2):
    """
    Renders the piecewise parametric equations onto a high-resolution canvas (>= 16 px/lat).
    Also returns the locations of lattice nodes.
    """
    params = AFFINE_PARAMS[family]
    bound = params['coord_bound']
    
    # Canvas dimension >= 16 px / lat
    canvas_dim = int(np.ceil((2 * bound) * scale_px_lat + 40))
    canvas = np.zeros((canvas_dim, canvas_dim), dtype=bool)
    origin = canvas_dim / 2.0
    
    # Rasterize each piece
    for p in pieces:
        p_type = p['type']
        if p_type == 'line':
            p0, p1 = np.array(p['p0']), np.array(p['p1'])
            t_vals = np.linspace(0, 1, max(5, int(np.linalg.norm(p1 - p0) * scale_px_lat * 2)))
            pts = p0 + np.outer(t_vals, (p1 - p0))
        elif p_type == 'arc':
            h, k, r = p['h'], p['k'], p['r']
            th0, th1 = p['th0'], p['th1']
            d_th = abs(th1 - th0)
            t_vals = np.linspace(th0, th1, max(5, int(r * d_th * scale_px_lat * 2)))
            pts = np.column_stack([h + r * np.cos(t_vals), k + r * np.sin(t_vals)])
        else: # bezier
            P0, P1, P2, P3 = np.array(p['P0']), np.array(p['P1']), np.array(p['P2']), np.array(p['P3'])
            t = np.linspace(0, 1, 25)[:, None]
            pts = (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t)*t**2 * P2 + t**3 * P3
            
        # Map to canvas pixels
        px = np.clip(np.round(origin + pts[:, 0] * scale_px_lat).astype(int), 0, canvas_dim - 1)
        py = np.clip(np.round(origin + pts[:, 1] * scale_px_lat).astype(int), 0, canvas_dim - 1)
        
        for i in range(len(px) - 1):
            rr, cc = line(py[i], px[i], py[i+1], px[i+1])
            valid = (rr >= 0) & (rr < canvas_dim) & (cc >= 0) & (cc < canvas_dim)
            canvas[rr[valid], cc[valid]] = True
            
    if stroke_px > 1:
        canvas_thick = binary_dilation(canvas, iterations=stroke_px // 2)
    else:
        canvas_thick = canvas
        
    return canvas_thick, canvas, canvas_dim, scale_px_lat, origin

def compute_reconstruction_metrics(img_path, path_logical, family='kolam19'):
    """
    Computes rigorous metrics between image stroke skeleton and rendered path:
    - Chamfer and Hausdorff in lattice units
    - F1 at 0.25 and 0.5 lattice-unit tolerance
    - Path-to-original Chamfer
    """
    im = Image.open(img_path).convert('L')
    arr = np.array(im)
    h, w = arr.shape
    
    # Original skeleton
    stroke_orig = remove_small_objects(arr < 200, max_size=16)
    skel_orig = skeletonize(stroke_orig)
    
    # Render path on image pixel grid
    pts_pix = logical_to_pixel(path_logical, family)
    rendered_skel = np.zeros((h, w), dtype=bool)
    for i in range(len(pts_pix) - 1):
        rr, cc = line(int(round(pts_pix[i, 1])), int(round(pts_pix[i, 0])),
                      int(round(pts_pix[i+1, 1])), int(round(pts_pix[i+1, 0])))
        valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rendered_skel[rr[valid], cc[valid]] = True
        
    scale_px_lat = AFFINE_PARAMS[family]['sx']
    
    dt_orig = distance_transform_edt(~skel_orig)
    dt_rend = distance_transform_edt(~rendered_skel)
    
    y_rend, x_rend = np.where(rendered_skel)
    y_orig, x_orig = np.where(skel_orig)
    
    d_rend_to_orig = dt_orig[y_rend, x_rend] / scale_px_lat
    d_orig_to_rend = dt_rend[y_orig, x_orig] / scale_px_lat
    
    chamfer = float(0.5 * (np.mean(d_rend_to_orig) + np.mean(d_orig_to_rend)))
    hausdorff = float(max(np.max(d_rend_to_orig), np.max(d_orig_to_rend)))
    
    f1_scores = {}
    for tol in [0.25, 0.5]:
        prec = float(np.mean(d_rend_to_orig <= tol))
        rec = float(np.mean(d_orig_to_rend <= tol))
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_scores[tol] = {'precision': prec, 'recall': rec, 'f1': f1}
        
    # Discrete path-to-original Chamfer
    px = np.clip(np.round(pts_pix[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(pts_pix[:, 1]).astype(int), 0, h - 1)
    path_to_orig_chamfer = float(np.mean(dt_orig[py, px]) / scale_px_lat)
    
    return {
        'chamfer_lat': chamfer,
        'hausdorff_lat': hausdorff,
        'f1_025': f1_scores[0.25]['f1'],
        'f1_050': f1_scores[0.50]['f1'],
        'path_to_orig_chamfer_lat': path_to_orig_chamfer,
        'skel_orig': skel_orig,
        'rendered_skel': rendered_skel
    }

def generate_3panel_figure(img_path, path_logical, family='kolam19', out_path=None):
    """
    Saves a 4-panel visual figure:
    1. Original Image
    2. Rendered from Equations (>= 16 px/lat canvas)
    3. Composite Overlay (Original in cyan, Rendered in magenta)
    4. Difference Map (Heatmap of spatial distance errors)
    """
    im = Image.open(img_path).convert('RGB')
    arr_orig = np.array(im)
    h, w, _ = arr_orig.shape
    
    metrics = compute_reconstruction_metrics(img_path, path_logical, family)
    skel_orig = metrics['skel_orig']
    rend_skel = metrics['rendered_skel']
    scale_px_lat = AFFINE_PARAMS[family]['sx']
    
    # Distance map of rendered skeleton to original
    dt_orig = distance_transform_edt(~skel_orig)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
    fig.patch.set_facecolor('#0b0b14')
    
    # 1. Original
    axes[0].set_facecolor('#0b0b14')
    axes[0].imshow(arr_orig)
    axes[0].set_title(f'1. Original Image ({family})', color='#ff9933', fontsize=11, weight='bold')
    axes[0].axis('off')
    
    # 2. Rendered from Equations
    ch_lat = metrics['chamfer_lat']
    hd_lat = metrics['hausdorff_lat']
    f1_val = metrics['f1_025'] * 100.0
    axes[1].set_facecolor('#0b0b14')
    axes[1].imshow(rend_skel, cmap='gray')
    axes[1].set_title(f'2. Rendered from Equations\n(Chamfer: {ch_lat:.3f} lat)', color='#ff9933', fontsize=11, weight='bold')
    axes[1].axis('off')
    
    # 3. Overlay (Cyan = Original, Magenta = Rendered)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    # Cyan: [0, 255, 255] for original skeleton
    overlay[skel_orig] = [0, 255, 255]
    # Magenta: [255, 0, 255] for rendered skeleton
    overlay[rend_skel] = np.maximum(overlay[rend_skel], [255, 0, 255])
    # Where both overlap -> White [255, 255, 255]
    overlap_mask = skel_orig & rend_skel
    overlay[overlap_mask] = [255, 255, 255]
    
    axes[2].set_facecolor('#0b0b14')
    axes[2].imshow(overlay)
    axes[2].set_title(f'3. Overlay (Cyan: Orig, Magenta: Rend)\nF1@0.25: {f1_val:.1f}%', color='#ff9933', fontsize=11, weight='bold')
    axes[2].axis('off')
    
    # 4. Difference Map (Error heatmap)
    diff_map = np.zeros((h, w), dtype=np.float32)
    diff_map[rend_skel] = dt_orig[rend_skel] / scale_px_lat
    
    axes[3].set_facecolor('#0b0b14')
    im_diff = axes[3].imshow(diff_map, cmap='magma', vmin=0, vmax=0.5)
    axes[3].set_title(f'4. Error Heatmap (Hausdorff: {hd_lat:.3f} lat)', color='#ff9933', fontsize=11, weight='bold')
    axes[3].axis('off')
    cbar = fig.colorbar(im_diff, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#f7f3e9')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#f7f3e9', fontsize=8)
    
    plt.suptitle(f'Reconstruction Quality: {family} vs Master Pair-1 Equations', color='#f7f3e9', fontsize=13, weight='bold')
    plt.tight_layout()
    
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, facecolor='#0b0b14')
    plt.close(fig)
    return metrics

def run_5image_deviation_table(zip_path="one-stroke-dotted-pulli-kolam.zip", images_dir="data/images"):
    """
    Evaluates 5 images per family and constructs a summary DataFrame of deviations.
    """
    table_rows = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for fam in ['kolam19', 'kolam29', 'kolam109']:
            csv_path = [n for n in z.namelist() if n.endswith(f'{fam}.csv')][0]
            df = pd.read_csv(io.BytesIO(z.read(csv_path)))
            p1 = df.iloc[:, 0:2].dropna().values
            
            for idx in range(5):
                img_f = os.path.join(images_dir, f'{fam}-{idx}.jpg')
                if os.path.exists(img_f):
                    m = compute_reconstruction_metrics(img_f, p1, fam)
                    table_rows.append({
                        'Family': fam,
                        'Image': f'{fam}-{idx}.jpg',
                        'Chamfer_lat': m['chamfer_lat'],
                        'Hausdorff_lat': m['hausdorff_lat'],
                        'F1_025_pct': m['f1_025'] * 100.0,
                        'F1_050_pct': m['f1_050'] * 100.0,
                        'Pass_025_tol': m['chamfer_lat'] <= 0.25 and m['f1_025'] >= 0.90
                    })
    return pd.DataFrame(table_rows)

if __name__ == '__main__':
    df = run_5image_deviation_table()
    print(df.to_string())
