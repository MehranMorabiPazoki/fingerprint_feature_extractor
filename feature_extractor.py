import cv2
import numpy as np
from skimage.morphology import skeletonize
import math

def preprocess_image(img):
    """Enhance and binarize 192x92 image."""
    if img is None:
        raise ValueError("Image is None.")
    # Apply CLAHE for better contrast in low-quality images
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # img = clahe.apply(img)
    # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = binary / 255.0
    thinned = skeletonize(binary).astype(np.uint8) * 255
    return thinned

def get_ridge_orientation(thinned, x, y, window_size=3):
    """Estimate orientation(s) with Sobel gradients."""
    try:
        patch = thinned[max(0, y-window_size):y+window_size+1, max(0, x-window_size):x+window_size+1]
        if patch.size == 0 or np.sum(patch) == 0:
            return np.nan
        sobelx = cv2.Sobel(patch.astype(float), cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(patch.astype(float), cv2.CV_64F, 0, 1, ksize=3)
        angle = math.atan2(np.sum(sobely), np.sum(sobelx)) * 180 / math.pi
        if np.sum(patch > 0) > 2:  # Bifurcation heuristic
            return [angle, (angle + 120) % 360 - 180 if (angle + 120) > 180 else (angle + 120) % 360,
                    (angle - 120) % 360 - 180 if (angle - 120) > 180 else (angle - 120) % 360]
        return angle
    except:
        return np.nan

def extract_minutiae(img):
    """Extract minutiae, handling scalar/list orientations."""
    
    thinned = preprocess_image(img)
    minutiae = []
    rows, cols = thinned.shape
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if thinned[i, j] == 255:
                neighbors = np.array([
                    thinned[i-1, j-1], thinned[i-1, j], thinned[i-1, j+1],
                    thinned[i, j+1], thinned[i+1, j+1], thinned[i+1, j],
                    thinned[i+1, j-1], thinned[i, j-1]
                ], dtype=np.uint8)
                transitions = np.sum(np.abs((neighbors // 255).astype(np.int8) - 
                                          np.roll(neighbors // 255, -1).astype(np.int8))) // 2
                orientation = get_ridge_orientation(thinned, j, i)
                if transitions == 1 and isinstance(orientation, float) and not np.isnan(orientation):
                    minutiae.append((np.int16(j), np.int16(i), orientation, 'Termination'))
                elif transitions == 3 and isinstance(orientation, list) and not np.any(np.isnan(orientation)):
                    minutiae.append((np.int16(j), np.int16(i), orientation, 'Bifurcation'))
    
    # Relaxed filtering for 192x92
    minutiae = [m for m in minutiae if 3 < m[0] < cols-3 and 3 < m[1] < rows-3]
    unique_minutiae = []
    for m in minutiae:
        if all(np.linalg.norm(np.array(m[:2]) - np.array(u[:2])) > 4 for u in unique_minutiae):
            unique_minutiae.append(m)
    
    # Debug: Save thinned image and print minutiae
    # cv2.imwrite('thinned.png', thinned)
    # print(f"Extracted {len(unique_minutiae)} minutiae: {unique_minutiae}")
    return unique_minutiae