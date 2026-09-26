import numpy as np
from skimage.morphology import skeletonize
from skimage import draw
from scipy.ndimage import label, convolve

def compute_betti_numbers(binary_image):
    """
    Extracts topological invariants (Betti numbers) from a binary skeleton.
    beta_0: Number of connected components
    beta_1: Number of independent loops (cycles) -> derived via Euler-Poincaré 
    """
    # Ensure 1-pixel thick skeleton via Zhang-Suen
    skeleton = skeletonize(binary_image > 0)
    
    # Calculate Beta 0 (Connected Components)
    _, beta_0 = label(skeleton, structure=np.ones((3,3)))
    
    # Calculate Beta 1 via Graph Euler Characteristic (Chi = V - E)
    # For a 1D curve skeleton, Beta_1 = E - V + Beta_0
    V = np.sum(skeleton)
    
    # Count 8-connected edges
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    neighbors = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    # Summing all neighbor connections and dividing by 2 (undirected graph edges)
    E = np.sum(neighbors[skeleton]) / 2.0
    
    beta_1 = int(max(0, E - V + beta_0))
    
    return beta_0, beta_1

def render_path_to_binary(x, y, img_size=128):
    """Rasterizes continuous mathematical pathways into a discrete binary matrix."""
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # Min-Max spatial normalization padding to prevent array bound truncation
    if x_max > x_min:
        x_norm = (x - x_min) / (x_max - x_min) * (img_size - 10) + 5
    else:
        x_norm = np.ones_like(x) * (img_size // 2)
        
    if y_max > y_min:
        y_norm = (y - y_min) / (y_max - y_min) * (img_size - 10) + 5
    else:
        y_norm = np.ones_like(y) * (img_size // 2)
        
    x_norm = np.round(x_norm).astype(int)
    y_norm = np.round(y_norm).astype(int)
    
    # Bresenham rasterization
    for i in range(len(x_norm) - 1):
        rr, cc = draw.line(y_norm[i], x_norm[i], y_norm[i+1], x_norm[i+1])
        valid = (rr >= 0) & (rr < img_size) & (cc >= 0) & (cc < img_size)
        img[rr[valid], cc[valid]] = 1
        
    return img

def evaluate_topology(skeleton_mask, x_spline, y_spline, x_fourier, y_fourier):
    """
    Executes a complete topological homology evaluation across all 3 pipeline phases.
    """
    orig_b0, orig_b1 = compute_betti_numbers(skeleton_mask)
    
    spline_bin = render_path_to_binary(x_spline, y_spline)
    sp_b0, sp_b1 = compute_betti_numbers(spline_bin)
    
    fourier_bin = render_path_to_binary(x_fourier, y_fourier)
    f_b0, f_b1 = compute_betti_numbers(fourier_bin)
    
    return {
        "original": (orig_b0, orig_b1),
        "spline": (sp_b0, sp_b1),
        "fourier": (f_b0, f_b1)
    }
