# enrollment.py
import sqlite3
import json
from feature_extractor import extract_minutiae
import cv2

def enroll_fingerprint(user_id, img, db_path='fingerprints.db'):
    """Enroll minutiae template."""
    minutiae = extract_minutiae(img)
    template = [(int(x), int(y), [float(a) for a in orient] if isinstance(orient, list) else float(orient), typ)
                for x, y, orient, typ in minutiae]
    template_json = json.dumps(template)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS templates (user_id TEXT PRIMARY KEY, minutiae BLOB)')
    cursor.execute('INSERT OR REPLACE INTO templates VALUES (?, ?)', (user_id, template_json.encode()))
    conn.commit()
    conn.close()
    print(f"Enrolled {user_id} with {len(minutiae)} minutiae")

# Enroll multiple images
import glob
socofing_files = glob.glob('dataset/archive/socofing/SOCOFing/Real/*.BMP')
for i, file in enumerate(socofing_files[:10]):
    
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {file}")
    print(f"file{file}")
    enroll_fingerprint(f'user{i+1}', img)