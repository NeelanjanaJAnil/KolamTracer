import numpy as np
from topology import render_path_to_binary, compute_betti_numbers
from spectral import choose_K

from scipy.ndimage import binary_dilation

def calculate_iou(bin1, bin2):
    """Calculates Intersection-over-Union between two binary masks with slight morphological tolerance."""
    # 1-pixel dilation accounts for Bresenham grid discretization
    d1 = binary_dilation(bin1 > 0, iterations=1)
    d2 = binary_dilation(bin2 > 0, iterations=1)
    intersection = np.logical_and(d1, d2).sum()
    union = np.logical_or(d1, d2).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)

def detect_symmetries(points, iou_thresh=0.95, img_size=128):
    """
    Detects mirror-x, mirror-y, rot90, and rot180 symmetries by rasterizing
    the path and measuring IoU under each geometric transform.
    
    Transforms (applied centered at the bounding box centroid):
    - mirror_x: (x, -y) or x-axis reflection (vertical flip)
    - mirror_y: (-x, y) or y-axis reflection (horizontal flip)
    - rot90: (-y, x) 90-degree counterclockwise rotation
    - rot180: (-x, -y) 180-degree rotation / inversion
    
    Returns:
        symmetry_results: dict mapping transform name to boolean & IoU value.
        group_name: Standard mathematical group name (e.g., 'C1', 'C2', 'C4', 'D1_x', 'D1_y', 'D2', 'D4')
    """
    pts = np.asarray(points, dtype=np.float64)
    # Center points at their centroid for geometric symmetry check
    centroid = np.mean(pts, axis=0)
    centered_pts = pts - centroid
    
    base_bin = render_path_to_binary(centered_pts[:, 0], centered_pts[:, 1], img_size=img_size)
    
    # 1. Mirror-X (flip across x-axis, i.e., y -> -y)
    mx_pts = np.column_stack([centered_pts[:, 0], -centered_pts[:, 1]])
    mx_bin = render_path_to_binary(mx_pts[:, 0], mx_pts[:, 1], img_size=img_size)
    iou_mx = calculate_iou(base_bin, mx_bin)
    
    # 2. Mirror-Y (flip across y-axis, i.e., x -> -x)
    my_pts = np.column_stack([-centered_pts[:, 0], centered_pts[:, 1]])
    my_bin = render_path_to_binary(my_pts[:, 0], my_pts[:, 1], img_size=img_size)
    iou_my = calculate_iou(base_bin, my_bin)
    
    # 3. Rot90 (counterclockwise 90 deg: x -> -y, y -> x)
    r90_pts = np.column_stack([-centered_pts[:, 1], centered_pts[:, 0]])
    r90_bin = render_path_to_binary(r90_pts[:, 0], r90_pts[:, 1], img_size=img_size)
    iou_r90 = calculate_iou(base_bin, r90_bin)
    
    # 4. Rot180 (180 deg: x -> -x, y -> -y)
    r180_pts = np.column_stack([-centered_pts[:, 0], -centered_pts[:, 1]])
    r180_bin = render_path_to_binary(r180_pts[:, 0], r180_pts[:, 1], img_size=img_size)
    iou_r180 = calculate_iou(base_bin, r180_bin)
    
    is_mx = iou_mx >= iou_thresh
    is_my = iou_my >= iou_thresh
    is_r90 = iou_r90 >= iou_thresh
    is_r180 = iou_r180 >= iou_thresh or is_r90 # rot90 implies rot180
    
    # Determine symmetry group
    if is_r90 and is_mx and is_my:
        group_name = "D4" # Dihedral group of order 8 (4 reflections + 4 rotations)
        order = 4
    elif is_r90:
        group_name = "C4" # Cyclic group of order 4 (90 deg rotations)
        order = 4
    elif is_r180 and is_mx and is_my:
        group_name = "D2" # Klein four-group / D2
        order = 2
    elif is_r180:
        group_name = "C2" # 180 deg rotational symmetry
        order = 2
    elif is_mx:
        group_name = "D1_x" # Reflection across x-axis
        order = 2
    elif is_my:
        group_name = "D1_y" # Reflection across y-axis
        order = 2
    else:
        group_name = "C1" # Trivial symmetry
        order = 1
        
    return {
        'group': group_name,
        'order': order,
        'mirror_x': {'detected': is_mx, 'iou': iou_mx},
        'mirror_y': {'detected': is_my, 'iou': iou_my},
        'rot90': {'detected': is_r90, 'iou': iou_r90},
        'rot180': {'detected': is_r180, 'iou': iou_r180},
    }

def get_fundamental_domain_path(points, group_name):
    """
    Extracts the fundamental domain path.
    For an N-fold rotational symmetry (e.g. C2, C4, D2, D4),
    the fundamental path corresponds to a 1/N segment of the curve.
    """
    pts = np.asarray(points, dtype=np.float64)
    N = len(pts)
    
    if group_name in ["C4", "D4"]:
        # 1/4 of the sequence
        domain_len = max(2, N // 4)
        fund_path = pts[:domain_len]
        reduction_factor = 4
    elif group_name in ["C2", "D2", "D1_x", "D1_y"]:
        # 1/2 of the sequence
        domain_len = max(2, N // 2)
        fund_path = pts[:domain_len]
        reduction_factor = 2
    else:
        fund_path = pts
        reduction_factor = 1
        
    return fund_path, reduction_factor

def regenerate_full_path_from_domain(fund_path, group_name, total_points=512):
    """
    Reconstructs the full periodic trajectory by applying group transformations.
    """
    pts = np.asarray(fund_path, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    
    if group_name in ["C4", "D4"]:
        # 4 rotations of 90 degrees
        full_pieces = [pts]
        for angle in [np.pi / 2, np.pi, 3 * np.pi / 2]:
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            rot = centered @ R.T + centroid
            full_pieces.append(rot)
        return np.vstack(full_pieces)
    elif group_name in ["C2", "D2"]:
        # 180 deg rotation
        rot180 = -centered + centroid
        return np.vstack([pts, rot180])
    elif group_name == "D1_x":
        flipped = np.column_stack([centered[:, 0], -centered[:, 1]]) + centroid
        return np.vstack([pts, flipped])
    elif group_name == "D1_y":
        flipped = np.column_stack([-centered[:, 0], centered[:, 1]]) + centroid
        return np.vstack([pts, flipped])
    else:
        return pts

def analyze_symmetry_reduction(points, gt_points=None, eps_lattice=0.25):
    """
    Analyzes whether symmetry allows reducing the equation count:
    - Counts equations with vs without symmetry.
    - Asserts that regenerated full curve passes validation before claiming reduction.
    """
    sym_info = detect_symmetries(points)
    group = sym_info['group']
    order = sym_info['order']
    
    # 1. Full curve without symmetry
    K_full, valid_full, metrics_full = choose_K(points, gt_points if gt_points is not None else points, eps_lattice=eps_lattice)
    eq_count_full = 2 * (2 * K_full + 1) # x and y real equations, or K harmonics
    
    if order == 1:
        return {
            'symmetry_group': group,
            'reduction_claimed': False,
            'equations_without_symmetry': eq_count_full,
            'equations_with_symmetry': eq_count_full,
            'reduction_ratio': 1.0,
            'regenerated_valid': valid_full,
            'K_used': K_full
        }
        
    # Fundamental domain path
    fund_path, factor = get_fundamental_domain_path(points, group)
    regen_full = regenerate_full_path_from_domain(fund_path, group)
    
    # Verify that the regenerated path matches topology & error
    gt_target = gt_points if gt_points is not None else points
    K_regen, valid_regen, metrics_regen = choose_K(regen_full, gt_target, eps_lattice=eps_lattice)
    
    if valid_regen and (metrics_regen['rms'] <= eps_lattice):
        # We can validly claim a reduction by factor
        eq_count_sym = max(2, eq_count_full // factor)
        return {
            'symmetry_group': group,
            'reduction_claimed': True,
            'equations_without_symmetry': eq_count_full,
            'equations_with_symmetry': eq_count_sym,
            'reduction_ratio': float(factor),
            'regenerated_valid': True,
            'K_used': K_full,
            'K_fundamental': K_regen
        }
    else:
        # Cannot claim reduction because regenerated full curve fails validation
        return {
            'symmetry_group': group,
            'reduction_claimed': False,
            'equations_without_symmetry': eq_count_full,
            'equations_with_symmetry': eq_count_full,
            'reduction_ratio': 1.0,
            'regenerated_valid': False,
            'K_used': K_full,
            'reason': "Regenerated curve did not pass topological/RMS validation."
        }
