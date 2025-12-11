
# Minutiae-Based Fingerprint Recognition System

This repository provides a complete implementation of a **minutiae-based fingerprint recognition system**, including:

- Fingerprint preprocessing  
- Skeletonization  
- Minutiae (ridge ending & bifurcation) detection  
- Orientation estimation  
- Minutiae filtering and deduplication  
- Polar coordinate transformation  
- Minutiae matching  
- Confidence score estimation  

The system is designed for **lightweight biometric applications**, embedded systems, and research experiments.

---

## 📘 Table of Contents
1. [Overview](#overview)
2. [Mathematical Foundations](#mathematical-foundations)
   - Skeletonization
   - Crossing Number Formula
   - Sobel Orientation
   - Polar Transform
   - Relative Orientation
   - Matching Constraints
   - Confidence Score
3. [System Pipeline](#system-pipeline)
4. [Module Descriptions](#module-descriptions)
5. [Usage Example](#usage-example)
6. [Output Format](#output-format)
7. [Limitations & Improvements](#limitations--improvements)
8. [License](#license)

---

# Overview

The system extracts minutiae (ridge endings and bifurcations) from a raw fingerprint image and performs matching using a **local polar transformation**–based approach. It is specifically tuned for small-resolution fingerprints such as **192×92 sensor images**.

---

# Mathematical Foundations

## 1. Skeletonization

Skeletonization reduces fingerprint ridges to 1‑pixel thickness while preserving ridge topology.  
Conditions to remove a pixel *P*:

\[
2 \le B(P_1..P_8) \le 6
\]
\[
A(P_1..P_8) = 1
\]

Where:

- \( B \) = number of non-zero neighbors  
- \( A \) = number of 0→1 transitions in the ordered 8-neighborhood  

This preserves connectivity of ridge lines.

---

## 2. Crossing Number Formula

Used to detect minutiae.

Neighbor ordering:

```
p1 p2 p3
p8 P  p4
p7 p6 p5
```

Let:

\[
N = [p_1, p_2, ..., p_8]
\]

Crossing Number:

\[
CN = rac{1}{2} \sum_{i=1}^{8} |N_i - N_{i+1}|,\quad N_9 = N_1
\]

Meaning:

| CN | Interpretation   | Type         |
|----|------------------|--------------|
| 1  | Ridge ending     | Termination  |
| 3  | Ridge splitting  | Bifurcation  |
| >3 | Noise            | Invalid      |

---

## 3. Sobel Orientation Estimation

Gradients:

\[
G_x = Sobel_x(patch), \quad G_y = Sobel_y(patch)
\]

Summed gradients:

\[
S_x = \sum G_x,\quad S_y = \sum G_y
\]

Orientation:

\[
	heta = 	an^{-1} \left( rac{S_y}{S_x} 
ight)
\]

For bifurcations (3 branches), estimated orientations:

\[
	heta,\; 	heta + 120^\circ,\; 	heta - 120^\circ
\]

---

## 4. Polar Coordinate Transform

For minutia \((x,y)\) relative to reference \((x_r,y_r)\):

\[
dx = x - x_r,\quad dy = y - y_r
\]

Distance:

\[
r = \sqrt{dx^2 + dy^2}
\]

Angle:

\[
\phi = 	an^{-1}\left(rac{dy}{dx}
ight)
\]

---

## 5. Relative Orientation

\[
	heta_{rel} = (	heta - 	heta_r + 360) \mod 360
\]

Provides **rotation invariance** for matching.

---

## 6. Matching Constraints

Two minutiae match if:

\[
|r_q - r_t| \le d_{th}
\]
\[
|\phi_q - \phi_t| \le \phi_{th}
\]
\[
|	heta_q - 	heta_t| \le 	heta_{th}
\]

And minutia type (ending/bifurcation) must match.

---

## 7. Confidence Score

\[
Score = rac{matched^2}{N_q \cdot N_t}
\]

Properties:

- Quadratic emphasis on stronger matches  
- Normalized to 0–1  
- Handles partial fingerprints gracefully  

---

# System Pipeline

```
Raw Image
    ↓
Gaussian Blurring
    ↓
Otsu Threshold → Binary Image
    ↓
Skeletonization
    ↓
Crossing Number for Minutiae Detection
    ↓
Sobel Gradient Orientation Estimation
    ↓
Boundary Filtering & Duplicate Removal
    ↓
---------------- Matching ----------------
    ↓
Centroid-based Reference Point Selection
    ↓
Polar Coordinate Transformation
    ↓
Threshold-Based Pairing
    ↓
Best Matching Score Computation
```

---

# Module Descriptions

## `fingerprint_feature.py`

### Key Functions  

#### `preprocess_image(img)`
- Gaussian blur  
- Otsu binarization (inverted for skeletonization)  
- Skeletonization  
- Returns 1‑pixel ridge image  

#### `get_ridge_orientation(thinned, x, y)`
- Computes Sobel gradients  
- Generates 1 orientation for terminations  
- Generates 3 orientations for bifurcations  

#### `extract_minutiae(img)`
- Crossing-number minutiae detection  
- Removes boundary minutiae  
- Removes duplicates within 4‑pixel radius  
- Output: `[(x, y, orientation, type), ...]`

---

## `fingerprint_matcher.py`

### Key Functions

#### `to_polar(minutiae, ref_idx)`
Transforms all minutiae into polar coordinates relative to a reference.

#### `match_polar(q_polar, t_polar)`
Greedy matching with thresholds for:

- distance  
- angular difference  
- orientation difference  
- type match  

#### `compute_confidence(query_minutiae, template_minutiae)`
- Selects 3 best reference points  
- Performs cross‑matching  
- Returns normalized score (0 to 1)

---

# Usage Example

```python
import cv2
from fingerprint_feature import extract_minutiae
from fingerprint_matcher import compute_confidence

img1 = cv2.imread("finger1.png", 0)
img2 = cv2.imread("finger2.png", 0)

m1 = extract_minutiae(img1)
m2 = extract_minutiae(img2)

score = compute_confidence(m1, m2)

print("Match score:", score)
```

---

# Output Format

Each extracted minutia is a tuple:

```
(x, y, orientation, type)
```

- `x, y` = pixel coordinates  
- `orientation` = float (ending) or 3‑element list (bifurcation)  
- `type` = `'Termination'` or `'Bifurcation'`  

---

# Limitations & Improvements

### ✔ Current Strengths
- Lightweight, fast  
- Works on small-resolution sensors  
- Orientation estimation included  
- Handles bifurcations with multi-angle representation  
- Rotation & translation invariant  

### ❗ Potential Improvements
- Replace Sobel orientation with Gabor-based orientation field  
- Add ridge frequency estimation  
- Replace greedy matcher with Hungarian algorithm  
- Add Hough-transform–based global alignment  
- Use deep learning for minutiae reliability scoring  

---

# License
MIT License  
Use freely for research, education, or industry.

