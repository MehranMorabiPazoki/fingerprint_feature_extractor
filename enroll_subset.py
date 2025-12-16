# enroll_subset.py
import cv2
import glob
import sqlite3
import json
import os
from feature_extractor import extract_minutiae

DB_PATH = "fingerprints.db"
DATASET_PATH = "SOKOTO/socofing/SOCOFing/Real"
MAX_SUBJECTS = 100        # <<<<< CHANGE THIS
MAX_FINGERS = 1          # <<<<< CHANGE THIS

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS templates (
            subject_id TEXT,
            finger_id TEXT,
            minutiae BLOB,
            PRIMARY KEY (subject_id, finger_id)
        )
    """)
    conn.commit()
    conn.close()

def parse_socofing_name(path):
    """
    Parses SOCOFing filename.

    Returns:
        subject_id (str)
        finger_id  (str)  e.g. 'Left_index', 'Right_thumb'
    """
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]

    # Example: 1__M_Left_index_finger
    subject_part, rest = name.split('__', 1)
    parts = rest.split('_')

    subject_id = subject_part
    hand = parts[1]          # Left / Right
    finger = parts[2]        # thumb / index / middle / ring / little

    finger_id = f"{hand}_{finger}"
    return subject_id, finger_id

def enroll():
    init_db()
    files = sorted(glob.glob(f"{DATASET_PATH}/*.BMP"))

    enrolled = {}

    for file in files:
        subject, finger = parse_socofing_name(file)

        if subject not in enrolled and len(enrolled) >= MAX_SUBJECTS:
            break

        enrolled.setdefault(subject, set())

        if len(enrolled[subject]) >= MAX_FINGERS:
            continue

        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        minutiae = extract_minutiae(img)
        if len(minutiae) < 5:
            continue

        minutiae = [
            (int(x), int(y),
             [float(a) for a in orient] if isinstance(orient, list) else float(orient),
             typ)
            for x, y, orient, typ in minutiae
        ]

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO templates VALUES (?, ?, ?)",
            (subject, finger, json.dumps(minutiae).encode())
        )
        conn.commit()
        conn.close()

        enrolled[subject].add(finger)
        print(f"Enrolled subject {subject}, finger {finger}")


if __name__ == "__main__":
    enroll()
