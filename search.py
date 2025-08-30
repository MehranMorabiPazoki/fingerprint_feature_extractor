import sqlite3
import json
from feature_extractor import extract_minutiae
from matcher import compute_confidence
import cv2


def search_database(img, db_path='fingerprints.db', conf_threshold=0.3):
    """Extract, match, return best ID and confidence."""
    try:
        
        query_minutiae = extract_minutiae(img)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None, 0.0
    
    if not query_minutiae:
        print("No minutiae extracted from query.")
        return None, 0.0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, minutiae FROM templates')
    results = []
    max_conf = 0
    for user_id, template_blob in cursor.fetchall():
        try:
            template_minutiae = json.loads(template_blob.decode())
            conf = compute_confidence(query_minutiae, template_minutiae,dist_thresh=10,angle_thresh=30)
            max_conf = max_conf if conf < max_conf else conf
            if conf >= conf_threshold:
                results.append((user_id, conf))
        except Exception as e:
            print(f"Error matching {user_id}: {e}")
    
    conn.close()
    
    if not results:
        return None, max_conf
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[0][0], results[0][1]

if __name__ == "__main__":
    query_path = 'dataset/archive/socofing/SOCOFing/Altered/Altered-Easy/543__M_Left_index_finger_Zcut.BMP'
    img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
            raise ValueError(f"Failed to load image: {query_path}")
    
    best_id, confidence = search_database(img)
    if best_id:
        print(f"Best match: {best_id} with confidence {confidence:.2f}")
    else:
        print(f"No match found above threshold. max confidence was {confidence}")