import sqlite3
import json

def load_templates(db_path='fingerprints.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, minutiae FROM templates")
    data = cursor.fetchall()
    conn.close()

    templates = {}
    for user_id, blob in data:
        templates[user_id] = json.loads(blob.decode())
    return templates

import cv2
import glob
import numpy as np
from feature_extractor import extract_minutiae
from matcher import compute_confidence   # your matcher file

def identify(query_minutiae, templates):
    scores = []
    for user_id, tmpl_minutiae in templates.items():
        score = compute_confidence(query_minutiae, tmpl_minutiae)
        scores.append((user_id, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def evaluate_identification(probe_files, templates, rank_k=[1, 5, 10]):
    correct_at_k = {k: 0 for k in rank_k}
    total = 0

    for file in probe_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Extract true identity from filename
        # Example: "001_3.bmp" → user001
        true_id = file.split('/')[-1].split('_')[0]

        query_minutiae = extract_minutiae(img)
        if len(query_minutiae) < 5:
            continue

        ranked = identify(query_minutiae, templates)
        total += 1

        for k in rank_k:
            top_k = [r[0] for r in ranked[:k]]
            if true_id in top_k:
                correct_at_k[k] += 1

    accuracy = {k: correct_at_k[k] / total for k in rank_k}
    return accuracy, total


templates = load_templates('fingerprints.db')

probe_files = glob.glob('SOCOFing/Real/*.BMP')

acc, total = evaluate_identification(probe_files, templates)

print(f"Total probes: {total}")
for k, v in acc.items():
    print(f"Rank-{k} Accuracy: {v*100:.2f}%")





# for attack in ['Obliteration', 'Rotation', 'Zcut']:
#     probe_files = glob.glob(f'SOCOFing/Altered/{attack}/*.BMP')
#     acc, total = evaluate_identification(probe_files, templates)

#     print(f"\nAttack: {attack}")
#     print(f"Total probes: {total}")
#     for k, v in acc.items():
#         print(f"Rank-{k}: {v*100:.2f}%")





# def compute_scores(probe_files, templates):
#     genuine, impostor = [], []

#     for file in probe_files:
#         img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             continue

#         true_id = file.split('/')[-1].split('_')[0]
#         q = extract_minutiae(img)

#         for uid, t in templates.items():
#             score = compute_confidence(q, t)
#             if uid == true_id:
#                 genuine.append(score)
#             else:
#                 impostor.append(score)

#     return genuine, impostor
