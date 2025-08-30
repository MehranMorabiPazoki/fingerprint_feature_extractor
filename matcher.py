import numpy as np
import math

def to_polar(minutiae, ref_idx):
    """Convert to polar coordinates."""
    ref_x, ref_y, ref_orient, _ = minutiae[ref_idx]
    ref_theta = ref_orient[0] if isinstance(ref_orient, list) else ref_orient
    polar = []
    for i, (x, y, orient, typ) in enumerate(minutiae):
        if i == ref_idx:
            continue
        dx, dy = x - ref_x, y - ref_y
        r = math.sqrt(dx**2 + dy**2)
        phi = math.atan2(dy, dx) * 180 / math.pi
        theta = orient[0] if isinstance(orient, list) else orient
        theta_rel = (theta - ref_theta + 360) % 360
        polar.append((r, phi, theta_rel, typ))
    return polar

def match_polar(q_polar, t_polar, dist_thresh=12, angle_thresh=25):
    """Pair minutiae."""
    matched = 0
    used = set()
    for q_r, q_phi, q_theta, q_typ in q_polar:
        for t_idx, (t_r, t_phi, t_theta, t_typ) in enumerate(t_polar):
            if t_idx in used:
                continue
            if abs(q_r - t_r) <= dist_thresh and abs(q_phi - t_phi) <= angle_thresh and abs(q_theta - t_theta) <= angle_thresh and q_typ == t_typ:
                matched += 1
                used.add(t_idx)
                break
    return matched

def compute_confidence(query_minutiae, template_minutiae, dist_thresh=15, angle_thresh=30):
    """Compute normalized confidence score."""
    if not query_minutiae or not template_minutiae:
        return 0.0
    
    q_centroid = np.mean([p[:2] for p in query_minutiae], axis=0)
    q_sorted = sorted(range(len(query_minutiae)), key=lambda i: np.linalg.norm(np.array(query_minutiae[i][:2]) - q_centroid))
    
    t_centroid = np.mean([p[:2] for p in template_minutiae], axis=0)
    t_sorted = sorted(range(len(template_minutiae)), key=lambda i: np.linalg.norm(np.array(template_minutiae[i][:2]) - t_centroid))
    
    best_matched = 0
    for q_ref in q_sorted[:3]:
        q_polar = to_polar(query_minutiae, q_ref)
        for t_ref in t_sorted[:3]:
            t_polar = to_polar(template_minutiae, t_ref)
            matched = match_polar(q_polar, t_polar, dist_thresh, angle_thresh)
            best_matched = max(best_matched, matched)
    
    score = (best_matched ** 2) / (len(query_minutiae) * len(template_minutiae)) if query_minutiae and template_minutiae else 0
    return min(score, 1.0)