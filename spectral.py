import numpy as np
import re
from topology import render_path_to_binary, compute_betti_numbers

def resample_uniform_arclength(points, n=512):
    """
    Resamples a 2D curve `points` (N, 2) to `n` points uniformly spaced by arc length.
    Ensures periodic closure if it is a loop.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.repeat(pts, n, axis=0)
    
    # Check if closed or close enough; if not already closed, close it smoothly
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-4:
        pts = np.vstack([pts, pts[0]])
        
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total_length = cum_dist[-1]
    
    if total_length == 0 or np.isnan(total_length):
        return np.repeat(pts[:1], n, axis=0)
    
    # Target uniform distances along the curve [0, total_length)
    target_dists = np.linspace(0.0, total_length, n, endpoint=False)
    
    x_interp = np.interp(target_dists, cum_dist, pts[:, 0])
    y_interp = np.interp(target_dists, cum_dist, pts[:, 1])
    
    return np.column_stack([x_interp, y_interp])


def fit_fourier(points, K):
    """
    Fits Fourier descriptors to a 2D curve of points (N, 2).
    First resamples to n=512 points uniformly by arc length.
    Computes complex coefficients c_k for k in [-K, K] via FFT.
    Returns:
        coeffs: dict mapping integer k in [-K, K] to complex c_k.
    """
    pts_resampled = resample_uniform_arclength(points, n=512)
    N = len(pts_resampled)
    z = pts_resampled[:, 0] + 1j * pts_resampled[:, 1]
    
    # Standard DFT
    Z = np.fft.fft(z) / N
    
    coeffs = {}
    for k in range(-K, K + 1):
        idx = k % N
        coeffs[k] = complex(Z[idx])
        
    return coeffs


def eval_fourier(coeffs, t):
    """
    Evaluates Fourier series at angles t in [0, 2*pi).
    z(t) = sum_{k=-K}^K c_k * e^{i * k * t}
    Returns: (x, y) coordinates array of shape (len(t), 2) or tuple of arrays.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    z = np.zeros_like(t_arr, dtype=np.complex128)
    
    for k, c_k in coeffs.items():
        z += c_k * np.exp(1j * k * t_arr)
        
    return np.real(z), np.imag(z)


def choose_K(points, gt_points, eps_lattice=0.25, k_max=80):
    """
    Finds the smallest K in [1, k_max] such that:
    1. RMS error <= eps_lattice (in lattice units, where 1 lattice unit = 1.0)
    2. beta0, beta1 match the GT path (rasterized using the exact same function).
    If none satisfy both, returns (best_K, False, best_metrics).
    """
    gt_pts = np.asarray(gt_points, dtype=np.float64)
    # Rasterize GT path
    gt_bin = render_path_to_binary(gt_pts[:, 0], gt_pts[:, 1], img_size=128)
    gt_b0, gt_b1 = compute_betti_numbers(gt_bin)
    
    # Compute GT arc-length resample for reference error calculation
    gt_resampled = resample_uniform_arclength(gt_pts, n=512)
    
    t_eval = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    
    best_K = 1
    min_rms = float('inf')
    best_metrics = {}
    
    for K in range(1, k_max + 1):
        coeffs = fit_fourier(points, K)
        fx, fy = eval_fourier(coeffs, t_eval)
        f_pts = np.column_stack([fx, fy])
        
        # RMS error against GT uniform path
        rms = float(np.sqrt(np.mean(np.sum((f_pts - gt_resampled) ** 2, axis=1))))
        
        # Topology of reconstructed Fourier curve
        f_bin = render_path_to_binary(fx, fy, img_size=128)
        fb0, fb1 = compute_betti_numbers(f_bin)
        
        if rms < min_rms:
            min_rms = rms
            best_K = K
            best_metrics = {
                'K': K,
                'rms': rms,
                'beta0': fb0,
                'beta1': fb1,
                'gt_beta0': gt_b0,
                'gt_beta1': gt_b1,
                'match': (rms <= eps_lattice and fb0 == gt_b0 and fb1 == gt_b1)
            }
            
        if rms <= eps_lattice and fb0 == gt_b0 and fb1 == gt_b1:
            return K, True, {
                'K': K,
                'rms': rms,
                'beta0': fb0,
                'beta1': fb1,
                'gt_beta0': gt_b0,
                'gt_beta1': gt_b1,
                'match': True
            }
            
    return best_K, False, best_metrics


def to_latex(coeffs, precision=3):
    """
    Produces:
    1. Single complex equation: z(t) = sum c_k e^{ikt}
    2. Coefficient table
    3. Merged real form:
       x(t) = a0_x + sum_{k=1}^K (a_k_x cos(kt) + b_k_x sin(kt))
       y(t) = a0_y + sum_{k=1}^K (a_k_y cos(kt) + b_k_y sin(kt))
       with no negative harmonics and clean signs (no double negatives).
    """
    sorted_k = sorted(coeffs.keys())
    
    # 1. Complex formula
    complex_terms = []
    for k in sorted_k:
        c = coeffs[k]
        cr = round(c.real, precision)
        ci = round(c.imag, precision)
        if abs(cr) < 1e-6 and abs(ci) < 1e-6:
            continue
        
        # formatting (cr + ci*i)
        if abs(ci) < 1e-6:
            c_str = f"{cr:.{precision}f}"
        elif abs(cr) < 1e-6:
            c_str = f"{ci:.{precision}f}i"
        elif ci >= 0:
            c_str = f"({cr:.{precision}f} + {ci:.{precision}f}i)"
        else:
            c_str = f"({cr:.{precision}f} - {abs(ci):.{precision}f}i)"
            
        if k == 0:
            complex_terms.append(f"{c_str}")
        elif k == 1:
            complex_terms.append(f"{c_str}e^{{it}}")
        elif k == -1:
            complex_terms.append(f"{c_str}e^{{-it}}")
        else:
            complex_terms.append(f"{c_str}e^{{{k}it}}")
            
    if not complex_terms:
        complex_latex = "z(t) = 0"
    else:
        complex_latex = "z(t) = " + " + ".join(complex_terms)
        # clean any "+ -"
        complex_latex = complex_latex.replace("+ -", "- ")
        
    # 2. Coefficient table
    table_rows = [
        "| Harmonic $k$ | Real Part $\\text{Re}(c_k)$ | Imag Part $\\text{Im}(c_k)$ | Amplitude $|c_k|$ | Phase $\\arg(c_k)$ |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]
    for k in sorted_k:
        c = coeffs[k]
        amp = abs(c)
        phase = np.angle(c)
        table_rows.append(
            f"| {k} | {c.real:.{precision}f} | {c.imag:.{precision}f} | {amp:.{precision}f} | {phase:.{precision}f} |"
        )
    coeff_table = "\n".join(table_rows)
    
    # 3. Merged real form
    # z(t) = c0 + sum_{k=1}^K [ c_k e^{ikt} + c_{-k} e^{-ikt} ]
    # c_k = (u_k + i v_k), c_{-k} = (u_{-k} + i v_{-k})
    # z(t) = (x(t) + i y(t))
    # For each k >= 1:
    # term_k = (u_k + i v_k)(cos kt + i sin kt) + (u_{-k} + i v_{-k})(cos kt - i sin kt)
    # real part: (u_k + u_{-k}) cos kt - (v_k - v_{-k}) sin kt
    # imag part: (v_k + v_{-k}) cos kt + (u_k - u_{-k}) sin kt
    K_max = max(abs(k) for k in coeffs.keys()) if coeffs else 0
    c0 = coeffs.get(0, 0.0 + 0.0j)
    
    a0_x = round(c0.real, precision)
    a0_y = round(c0.imag, precision)
    
    x_terms = [f"{a0_x:.{precision}f}"]
    y_terms = [f"{a0_y:.{precision}f}"]
    
    for k in range(1, K_max + 1):
        ck = coeffs.get(k, 0.0 + 0.0j)
        c_neg_k = coeffs.get(-k, 0.0 + 0.0j)
        
        u_k, v_k = ck.real, ck.imag
        u_nk, v_nk = c_neg_k.real, c_neg_k.imag
        
        # x(t) coeffs:
        ak_x = round(u_k + u_nk, precision)
        bk_x = round(- (v_k - v_nk), precision)
        
        # y(t) coeffs:
        ak_y = round(v_k + v_nk, precision)
        bk_y = round(u_k - u_nk, precision)
        
        # Append clean formatted terms for x(t)
        if abs(ak_x) > 1e-6:
            sign = " + " if ak_x >= 0 else " - "
            x_terms.append(f"{sign}{abs(ak_x):.{precision}f} \\cos({k}t)")
        if abs(bk_x) > 1e-6:
            sign = " + " if bk_x >= 0 else " - "
            x_terms.append(f"{sign}{abs(bk_x):.{precision}f} \\sin({k}t)")
            
        # Append clean formatted terms for y(t)
        if abs(ak_y) > 1e-6:
            sign = " + " if ak_y >= 0 else " - "
            y_terms.append(f"{sign}{abs(ak_y):.{precision}f} \\cos({k}t)")
        if abs(bk_y) > 1e-6:
            sign = " + " if bk_y >= 0 else " - "
            y_terms.append(f"{sign}{abs(bk_y):.{precision}f} \\sin({k}t)")
            
    x_eq = "x(t) = " + "".join(x_terms).strip()
    y_eq = "y(t) = " + "".join(y_terms).strip()
    
    # Clean leading "+ " if any
    x_eq = re.sub(r'=\s*\+\s*', '= ', x_eq)
    y_eq = re.sub(r'=\s*\+\s*', '= ', y_eq)
    
    real_latex = f"\\begin{{aligned}}\n{x_eq} \\\\\n{y_eq}\n\\end{{aligned}}"
    
    return {
        'complex_latex': complex_latex,
        'coeff_table': coeff_table,
        'real_latex': real_latex,
        'x_equation': x_eq,
        'y_equation': y_eq
    }


def parse_rounded_coeffs(coeff_table_text):
    """
    Parses back the rounded coefficients from the markdown table.
    Used for regeneration test assertion.
    """
    parsed = {}
    lines = coeff_table_text.strip().splitlines()
    for line in lines:
        if line.startswith("|") and not line.startswith("| Harmonic") and not line.startswith("| :---"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 3:
                k = int(parts[0])
                re_val = float(parts[1])
                im_val = float(parts[2])
                parsed[k] = complex(re_val, im_val)
    return parsed
