# evaluate_altered.py
import cv2
import glob
import json
import sqlite3
import os
from feature_extractor import extract_minutiae
from matcher import compute_confidence

DB_PATH = "fingerprints.db"
ALTERED_PATH = "SOKOTO/socofing/SOCOFing/Altered/Altered-Easy"


# ---------- Parser ----------
def parse_socofing_name(path):
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]

    subject_part, rest = name.split('__', 1)
    parts = rest.split('_')

    subject_id = subject_part
    hand = parts[1]
    finger = parts[2]
    finger_id = f"{hand}_{finger}"

    attack = parts[-1] if parts[-1] in ('CR', 'Obl', 'Zcut') else None
    return subject_id, finger_id, attack


# ---------- Load gallery ----------
def load_templates():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT subject_id, finger_id, minutiae FROM templates")
    rows = c.fetchall()
    conn.close()

    templates = {}
    for subject, finger, blob in rows:
        templates[(subject, finger)] = json.loads(blob.decode())
    return templates


# ---------- Identification ----------
def identify(query_minutiae, templates):
    scores = []
    for (subject, finger), tmpl in templates.items():
        score = compute_confidence(query_minutiae, tmpl)
        scores.append(((subject, finger), score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ---------- Evaluation ----------
def evaluate_altered(rank_k=(1, 5,10)):
    templates = load_templates()

    for attack in ['CR', 'Obl', 'Zcut']:
        files = glob.glob(f"{ALTERED_PATH}/*_{attack}.BMP")

        correct = {k: 0 for k in rank_k}
        total = 0

        for file in files:
            subject, finger, atk = parse_socofing_name(file)
            
            # Only evaluate fingers present in gallery
            if (subject, finger) not in templates:
                continue

            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            query = extract_minutiae(img)
            if len(query) < 5:
                continue

            ranked = identify(query, templates)
            total += 1

            for k in rank_k:
                top_k = [r[0] for r in ranked[:k]]
                if (subject, finger) in top_k:
                    correct[k] += 1

        print(f"\nAttack type: {attack}")
        print(f"Total probes: {total}")
        for k in rank_k:
            print(f"Rank-{k} Accuracy: {100 * correct[k] / total:.2f}%")


if __name__ == "__main__":
    evaluate_altered()
