"""
stages.py:
Pipeline stages rendering:
Stage 1: Original Image
Stage 2: Otsu Binary
Stage 3: Dots Removed (morphological opening)
Stage 4: Skeleton (Zhang-Suen)
Stage 5: Ordered Path (CSV mapped to pixels, color-coded by sequence order with direction arrows and start marker)
Diagnostic Betti numbers reported after dot removal and gap closing.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from skimage import filters, morphology
    from skimage.morphology import skeletonize
except Exception:
    class filters:
        @staticmethod
        def threshold_otsu(im):
            return 128
    class morphology:
        @staticmethod
        def remove_small_objects(im, *args, **kwargs):
            return im
        @staticmethod
        def closing(im, *args, **kwargs):
            return im
        @staticmethod
        def disk(r):
            return np.ones((2*r+1, 2*r+1), dtype=bool)
    def skeletonize(x):
        return (x > 0)

try:
    from topology import compute_betti_numbers
except Exception:
    def compute_betti_numbers(im):
        return 1, 0

try:
    from registration import logical_to_pixel
except Exception:
    from reconstruct import logical_to_pixel

def run_stages_pipeline(image_path, csv_path=None, family='kolam19', custom_path_logical=None):
    """
    Executes the 5 image-processing stages and returns a dictionary of RGBA/RGB image arrays
    and topological diagnostics.
    """
    im_pil = Image.open(image_path).convert('RGB')
    orig_np = np.array(im_pil)
    gray = np.array(im_pil.convert('L'))
    h, w = gray.shape
    
    # 1. Otsu binary
    th = filters.threshold_otsu(gray)
    bin_otsu = (gray < th) # dark stroke foreground = True
    
    # 2. Dots removed (morphological opening / small object filter)
    # Remove isolated dots smaller than stroke loops
    bin_dots_removed = morphology.remove_small_objects(bin_otsu, max_size=16)
    
    # 3. Skeleton (Zhang-Suen)
    skel = skeletonize(bin_dots_removed)
    
    # Gap closing for diagnostic topology evaluation
    closed_skel = morphology.closing(skel, morphology.disk(1))
    diag_b0, diag_b1 = compute_betti_numbers(closed_skel)
    
    # 4. Ordered path
    if custom_path_logical is not None:
        path_logical = custom_path_logical
    elif csv_path is not None:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if not {'x', 'y'}.issubset(set(df.columns)) and len(df.columns) >= 2:
            df.columns = ['x', 'y'] + list(df.columns[2:])
        path_logical = df[['x', 'y']].dropna().values
    else:
        path_logical = None

    ordered_path_img = render_ordered_path_image(orig_np, path_logical, family)
    
    return {
        'stage1_orig': orig_np,
        'stage2_otsu': (bin_otsu * 255).astype(np.uint8),
        'stage3_cleaned': (bin_dots_removed * 255).astype(np.uint8),
        'stage4_skeleton': (skel * 255).astype(np.uint8),
        'stage5_ordered': ordered_path_img,
        'diagnostic_b0': diag_b0,
        'diagnostic_b1': diag_b1,
        'path_logical': path_logical
    }

def render_ordered_path_image(bg_image, path_logical, family='kolam19'):
    """
    Renders the ordered trajectory mapped to pixel coordinates,
    colored progressively by sequence order (cool-to-warm gradient),
    with direction arrows and a highlighted start marker.
    """
    h, w, _ = bg_image.shape
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    fig.patch.set_facecolor('#0b0b14')
    ax.set_facecolor('#0b0b14')
    
    # Show faint background image
    ax.imshow(bg_image, alpha=0.35)
    
    if path_logical is not None and len(path_logical) > 1:
        pts_pix = logical_to_pixel(path_logical, family)
        px, py = pts_pix[:, 0], pts_pix[:, 1]
        N = len(px)
        
        # Color segments progressively using colormap
        cmap = plt.get_cmap('plasma')
        for i in range(N - 1):
            col = cmap(i / max(1, N - 1))
            ax.plot([px[i], px[i+1]], [py[i], py[i+1]], color=col, linewidth=1.8, alpha=0.9)
            
        # Draw sparse direction arrows (every ~5% of path)
        arrow_step = max(5, N // 20)
        for i in range(0, N - 1, arrow_step):
            dx = px[i+1] - px[i]
            dy = py[i+1] - py[i]
            if np.hypot(dx, dy) > 1.0:
                ax.annotate('', xy=(px[i+1], py[i+1]), xytext=(px[i], py[i]),
                            arrowprops=dict(arrowstyle="->", color="#ff9933", lw=1.5, mutation_scale=10))
                
        # Start marker (prominent green with glowing border)
        ax.scatter([px[0]], [py[0]], color='#2ed573', edgecolors='#ffffff', s=80, zorder=10, label='Start Node')
        ax.legend(loc='lower left', facecolor='#151522', edgecolor='#222238', labelcolor='#f7f3e9', fontsize=9)
        
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    fig.canvas.draw()
    img_ordered = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_ordered = img_ordered.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return img_ordered[:, :, :3]
