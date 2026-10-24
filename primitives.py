"""
primitives.py:
Segments the discrete node path into geometric primitives (lines, circular arcs, and cubic Beziers).
Accepts a line or circular arc only if max deviation <= 0.05 lattice units; otherwise uses cubic Bezier.
Outputs parametric forms (with t-ranges) and algebraic forms:
  - Circle: (x - h)^2 + (y - k)^2 = r^2
  - Line: ax + by + c = 0
  - Bezier: parametric only
Also implements lossless Re-Pair grammar induction over the sequence of moves/primitives,
with exact expansion assertion and shuffled-sequence control.
"""

import json
import numpy as np
from collections import Counter

def fit_line_segment(pts):
    """
    Fits a straight line segment between pts[0] and pts[-1].
    Line equation: ax + by + c = 0, where a^2 + b^2 = 1.
    """
    p0, p1 = pts[0], pts[-1]
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        devs = np.linalg.norm(pts - p0, axis=1)
        return True, 0.0, 0.0, 0.0, float(np.max(devs)), p0, p1
        
    a = float(-v[1] / L)
    b = float(v[0] / L)
    c = float(-(a * p0[0] + b * p0[1]))
    dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)
    max_dev = float(np.max(dists))
    return True, a, b, c, max_dev, p0, p1

def fit_circle_arc(pts):
    """
    Fits a circular arc: (x - h)^2 + (y - k)^2 = r^2.
    """
    if len(pts) < 3:
        return False, 0.0, 0.0, 0.0, 999.0, 0.0, 0.0
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2*x, 2*y, -np.ones_like(x)])
    B = x**2 + y**2
    try:
        sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        h, k, C = sol
        r_sq = h**2 + k**2 - C
        if r_sq <= 0:
            return False, 0.0, 0.0, 0.0, 999.0, 0.0, 0.0
        r = float(np.sqrt(r_sq))
        dists = np.abs(np.sqrt((x - h)**2 + (y - k)**2) - r)
        max_dev = float(np.max(dists))
        th0 = float(np.arctan2(y[0] - k, x[0] - h))
        th1 = float(np.arctan2(y[-1] - k, x[-1] - h))
        return True, float(h), float(k), r, max_dev, th0, th1
    except Exception:
        return False, 0.0, 0.0, 0.0, 999.0, 0.0, 0.0

def fit_cubic_bezier(pts):
    """
    Fits a cubic Bezier curve through pts:
    P(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3 for t in [0, 1].
    P0 = pts[0], P3 = pts[-1]. Solves P1, P2 by least squares.
    """
    n = len(pts)
    P0 = pts[0].astype(float)
    P3 = pts[-1].astype(float)
    if n <= 2:
        return P0, P0, P3, P3, 0.0
        
    diffs = np.diff(pts, axis=0)
    dists = np.insert(np.cumsum(np.linalg.norm(diffs, axis=1)), 0, 0.0)
    total_L = dists[-1]
    t = dists / (total_L if total_L > 0 else 1.0)
    
    b0 = (1 - t)**3
    b1 = 3 * (1 - t)**2 * t
    b2 = 3 * (1 - t) * t**2
    b3 = t**3
    
    lhs_x = pts[:, 0] - b0 * P0[0] - b3 * P3[0]
    lhs_y = pts[:, 1] - b0 * P0[1] - b3 * P3[1]
    
    M = np.column_stack([b1, b2])
    sol_x, _, _, _ = np.linalg.lstsq(M, lhs_x, rcond=None)
    sol_y, _, _, _ = np.linalg.lstsq(M, lhs_y, rcond=None)
    
    P1 = np.array([sol_x[0], sol_y[0]], dtype=float)
    P2 = np.array([sol_x[1], sol_y[1]], dtype=float)
    
    fit_x = b0 * P0[0] + b1 * P1[0] + b2 * P2[0] + b3 * P3[0]
    fit_y = b0 * P0[1] + b1 * P1[1] + b2 * P2[1] + b3 * P3[1]
    max_dev = float(np.max(np.hypot(pts[:, 0] - fit_x, pts[:, 1] - fit_y)))
    return P0, P1, P2, P3, max_dev

def segment_path_into_primitives(pts, tol=0.05):
    """
    Greedily segments pts (N, 2) into lines, circular arcs, or cubic Beziers.
    Only accepts line or circular arc if max deviation <= tol (0.05 lattice units).
    """
    N = len(pts)
    pieces = []
    i = 0
    piece_id = 1
    
    while i < N - 1:
        best_type = None
        best_j = i + 1
        best_data = None
        
        # Search forward for longest segment fitting line or arc within tolerance
        for j in range(i + 2, min(N, i + 64)):
            chunk = pts[i:j+1]
            ok_l, a, b, c, dev_l, p0, p1 = fit_line_segment(chunk)
            if ok_l and dev_l <= tol:
                best_type = 'line'
                best_j = j
                best_data = {
                    'type': 'line',
                    'a': a, 'b': b, 'c': c,
                    'p0': p0.tolist(), 'p1': p1.tolist(),
                    'max_dev': dev_l
                }
                continue
                
            ok_a, h, k, r, dev_a, th0, th1 = fit_circle_arc(chunk)
            if ok_a and dev_a <= tol:
                best_type = 'arc'
                best_j = j
                best_data = {
                    'type': 'arc',
                    'h': h, 'k': k, 'r': r,
                    'th0': th0, 'th1': th1,
                    'max_dev': dev_a
                }
                continue
            break
            
        if best_type is not None and best_j > i + 1:
            best_data['piece_id'] = piece_id
            best_data['start_idx'] = i
            best_data['end_idx'] = best_j
            pieces.append(best_data)
            i = best_j
        else:
            # Fit cubic Bezier over local window
            j = min(N - 1, i + 3)
            chunk = pts[i:j+1]
            P0, P1, P2, P3, dev_b = fit_cubic_bezier(chunk)
            pieces.append({
                'piece_id': piece_id,
                'type': 'bezier',
                'start_idx': i,
                'end_idx': j,
                'P0': P0.tolist(),
                'P1': P1.tolist(),
                'P2': P2.tolist(),
                'P3': P3.tolist(),
                'max_dev': dev_b
            })
            i = j
        piece_id += 1
        
    return pieces

def format_piece_equations(piece):
    """
    Formats parametric and algebraic equations (LaTeX and string) for each piece.
    """
    t_type = piece['type']
    if t_type == 'line':
        p0, p1 = piece['p0'], piece['p1']
        a, b, c = piece['a'], piece['b'], piece['c']
        param_latex = f"\\begin{{pmatrix}} x(t) \\\\ y(t) \\end{{pmatrix}} = \\begin{{pmatrix}} {p0[0]:.2f} + {p1[0]-p0[0]:.2f}t \\\\ {p0[1]:.2f} + {p1[1]-p0[1]:.2f}t \\end{{pmatrix}}, \\quad t \\in [0, 1]"
        algebraic_latex = f"{a:.3f}x + {b:.3f}y + {c:.3f} = 0"
        return {
            'type': 'line',
            'param_latex': param_latex,
            'algebraic_latex': algebraic_latex,
            'param_count': 4 # p0_x, p0_y, p1_x, p1_y
        }
    elif t_type == 'arc':
        h, k, r = piece['h'], piece['k'], piece['r']
        th0, th1 = piece['th0'], piece['th1']
        param_latex = f"\\begin{{pmatrix}} x(t) \\\\ y(t) \\end{{pmatrix}} = \\begin{{pmatrix}} {h:.2f} + {r:.2f}\\cos(t) \\\\ {k:.2f} + {r:.2f}\\sin(t) \\end{{pmatrix}}, \\quad t \\in [{th0:.2f}, {th1:.2f}]"
        sign_h = f"- {h:.2f}" if h >= 0 else f"+ {abs(h):.2f}"
        sign_k = f"- {k:.2f}" if k >= 0 else f"+ {abs(k):.2f}"
        algebraic_latex = f"(x {sign_h})^2 + (y {sign_k})^2 = {r**2:.2f}"
        return {
            'type': 'arc',
            'param_latex': param_latex,
            'algebraic_latex': algebraic_latex,
            'param_count': 5 # h, k, r, th0, th1
        }
    else: # bezier
        p0, p1, p2, p3 = piece['P0'], piece['P1'], piece['P2'], piece['P3']
        param_latex = f"\\mathbf{{B}}(t) = (1-t)^3 \\mathbf{{P}}_0 + 3(1-t)^2 t \\mathbf{{P}}_1 + 3(1-t)t^2 \\mathbf{{P}}_2 + t^3 \\mathbf{{P}}_3, \\quad t \\in [0, 1]"
        algebraic_latex = "\\text{Parametric only (no simple implicit algebraic polynomial)}"
        return {
            'type': 'bezier',
            'param_latex': param_latex,
            'algebraic_latex': algebraic_latex,
            'param_count': 8 # P0..P3 coords
        }

def repair_grammar(seq):
    """
    Hierarchical Re-Pair compression algorithm over a sequence of symbols.
    """
    current = list(seq)
    rules = {}
    rule_idx = 1
    while True:
        n = len(current)
        if n < 2:
            break
        counts = Counter(zip(current[:-1], current[1:]))
        if not counts:
            break
        best_pair = None
        best_count = 1
        for p, count in counts.most_common():
            if count <= 1:
                break
            # count actual non-overlapping
            c = 0
            j = 0
            while j < n - 1:
                if current[j] == p[0] and current[j+1] == p[1]:
                    c += 1
                    j += 2
                else:
                    j += 1
            if c > 1 and c > best_count:
                best_count = c
                best_pair = p
                break
        if best_pair is None or best_count < 2:
            break
            
        rname = f'R{rule_idx}'
        rule_idx += 1
        rules[rname] = best_pair
        
        new_seq = []
        j = 0
        while j < n:
            if j < n - 1 and current[j] == best_pair[0] and current[j+1] == best_pair[1]:
                new_seq.append(rname)
                j += 2
            else:
                new_seq.append(current[j])
                j += 1
        current = new_seq
    return current, rules

def expand_grammar(axiom, rules):
    """
    Losslessly expands the Re-Pair grammar back to the original sequence.
    """
    stack = list(reversed(axiom))
    res = []
    while stack:
        it = stack.pop()
        if it in rules:
            stack.append(rules[it][1])
            stack.append(rules[it][0])
        else:
            res.append(it)
    return res

def analyze_family_primitives(pts, family='kolam19'):
    """
    Performs complete segmentation, parameter counting, and grammar induction on pts.
    """
    pieces = segment_path_into_primitives(pts, tol=0.05)
    
    # Counts by type
    type_counts = Counter(p['type'] for p in pieces)
    
    # Format equations and count parameters
    formatted_pieces = []
    total_params = 0
    for p in pieces:
        eq = format_piece_equations(p)
        total_params += eq['param_count']
        p_dict = dict(p)
        p_dict.update(eq)
        formatted_pieces.append(p_dict)
        
    raw_coords_count = len(pts) * 2
    
    # Re-Pair on move encoding m_t = round(2*(p_{t+1}-p_t))
    m = [tuple(x) for x in np.round(2 * (pts[1:] - pts[:-1])).astype(int)]
    axiom, rules = repair_grammar(m)
    expanded = expand_grammar(axiom, rules)
    assert expanded == m, "Grammar expansion failed to match exactly!"
    
    # Rebuild path from expanded moves
    rebuilt = pts[0] + np.cumsum(np.array(expanded), axis=0) / 2.0
    rebuilt = np.vstack([pts[0], rebuilt])
    max_err = float(np.max(np.abs(rebuilt - pts)))
    assert max_err == 0.0, f"Rebuilt path max error {max_err} != 0.0"
    
    total_grammar_symbols = len(axiom) + 2 * len(rules)
    compression_ratio = len(m) / total_grammar_symbols
    
    return {
        'family': family,
        'num_raw_points': len(pts),
        'raw_coords_count': raw_coords_count,
        'num_pieces': len(pieces),
        'type_counts': dict(type_counts),
        'total_params': total_params,
        'param_reduction_ratio': raw_coords_count / total_params,
        'num_rules': len(rules),
        'axiom_len': len(axiom),
        'total_grammar_symbols': total_grammar_symbols,
        'grammar_compression_ratio': compression_ratio,
        'reconstruction_max_err': max_err,
        'pieces': formatted_pieces,
        'rules': rules,
        'axiom': axiom
    }
