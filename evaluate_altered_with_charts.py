# evaluate_altered_with_charts.py
import cv2
import glob
import json
import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from feature_extractor import extract_minutiae
from matcher import compute_confidence

DB_PATH = "fingerprints.db"
ALTERED_PATH = "SOKOTO/socofing/SOCOFing/Altered/Altered-Easy"
REPORT_DIR = "evaluation_reports"

# Create report directory
os.makedirs(REPORT_DIR, exist_ok=True)


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


# ---------- Plotting Functions ----------
def plot_rank_accuracy(results, rank_k):
    """Bar chart comparing rank accuracies across attack types."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    attacks = list(results.keys())
    x = np.arange(len(rank_k))
    width = 0.25
    
    for i, attack in enumerate(attacks):
        accuracies = [results[attack]['accuracy'][k] for k in rank_k]
        ax.bar(x + i * width, accuracies, width, label=attack)
    
    ax.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Rank-k Identification Accuracy by Attack Type', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Rank-{k}' for k in rank_k])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/rank_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cumulative_match_curve(results, rank_k):
    """CMC curve for each attack type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for attack, data in results.items():
        accuracies = [data['accuracy'][k] for k in rank_k]
        ax.plot(rank_k, accuracies, marker='o', linewidth=2, label=attack, markersize=8)
    
    ax.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Identification Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Match Characteristic (CMC) Curve', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xlim(rank_k[0], rank_k[-1])
    
    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/cmc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_distribution(all_scores):
    """Distribution of matching scores for genuine vs impostor comparisons."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for attack, scores in all_scores.items():
        genuine = scores['genuine']
        impostor = scores['impostor']
        
        if genuine:
            ax.hist(genuine, bins=30, alpha=0.5, label=f'{attack} - Genuine', density=True)
        if impostor:
            ax.hist(impostor, bins=30, alpha=0.5, label=f'{attack} - Impostor', density=True)
    
    ax.set_xlabel('Matching Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Score Distribution: Genuine vs Impostor Matches', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_attack_stats(results):
    """Statistics per attack type: total probes and average scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    attacks = list(results.keys())
    totals = [results[a]['total'] for a in attacks]
    avg_scores = [results[a]['avg_genuine_score'] for a in attacks]
    
    # Total probes
    bars1 = ax1.bar(attacks, totals, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax1.set_ylabel('Number of Probes', fontsize=12, fontweight='bold')
    ax1.set_title('Total Probe Images per Attack Type', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Average genuine scores
    bars2 = ax2.bar(attacks, avg_scores, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax2.set_ylabel('Average Matching Score', fontsize=12, fontweight='bold')
    ax2.set_title('Average Genuine Match Score per Attack Type', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/attack_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_heatmap(results, rank_k):
    """Heatmap showing rank-1 success rate breakdown."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    attacks = list(results.keys())
    k = rank_k[0]  # Use rank-1
    
    data = []
    for attack in attacks:
        correct = results[attack]['correct'][k]
        total = results[attack]['total']
        incorrect = total - correct
        data.append([correct, incorrect])
    
    data = np.array(data)
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Correct', 'Incorrect'], fontweight='bold')
    ax.set_yticks(np.arange(len(attacks)))
    ax.set_yticklabels(attacks, fontweight='bold')
    ax.set_title(f'Rank-{k} Identification Results Breakdown', 
                 fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(attacks)):
        for j in range(2):
            text = ax.text(j, i, int(data[i, j]),
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/rank1_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(results, rank_k):
    """Generate text summary report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(f'{REPORT_DIR}/evaluation_summary.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FINGERPRINT IDENTIFICATION EVALUATION REPORT\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 70 + "\n\n")
        
        for attack in results.keys():
            f.write(f"\n{'='*50}\n")
            f.write(f"Attack Type: {attack}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Total Probe Images: {results[attack]['total']}\n")
            f.write(f"Average Genuine Score: {results[attack]['avg_genuine_score']:.4f}\n\n")
            
            f.write("Rank-k Accuracy:\n")
            for k in rank_k:
                acc = results[attack]['accuracy'][k]
                correct = results[attack]['correct'][k]
                total = results[attack]['total']
                f.write(f"  Rank-{k:2d}: {acc:6.2f}% ({correct}/{total} correct)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 70 + "\n")
        
        total_probes = sum(r['total'] for r in results.values())
        f.write(f"Total Probes Across All Attacks: {total_probes}\n\n")
        
        for k in rank_k:
            avg_acc = np.mean([results[a]['accuracy'][k] for a in results.keys()])
            f.write(f"Average Rank-{k} Accuracy: {avg_acc:.2f}%\n")


# ---------- Evaluation with Data Collection ----------
def evaluate_altered(rank_k=(1, 5, 10)):
    templates = load_templates()
    
    results = {}
    all_scores = {}

    for attack in ['CR', 'Obl', 'Zcut']:
        files = glob.glob(f"{ALTERED_PATH}/*_{attack}.BMP")

        correct = {k: 0 for k in rank_k}
        total = 0
        genuine_scores = []
        impostor_scores = []

        for file in files:
            subject, finger, atk = parse_socofing_name(file)
            
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

            # Collect scores
            for (subj, fing), score in ranked:
                if (subj, fing) == (subject, finger):
                    genuine_scores.append(score)
                else:
                    impostor_scores.append(score)

            # Check rank accuracy
            for k in rank_k:
                top_k = [r[0] for r in ranked[:k]]
                if (subject, finger) in top_k:
                    correct[k] += 1

        # Store results
        accuracy = {k: 100 * correct[k] / total if total > 0 else 0 for k in rank_k}
        results[attack] = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'avg_genuine_score': np.mean(genuine_scores) if genuine_scores else 0
        }
        
        all_scores[attack] = {
            'genuine': genuine_scores,
            'impostor': impostor_scores[:len(genuine_scores)*10]  # Subsample for visualization
        }

        # Print to console
        print(f"\nAttack type: {attack}")
        print(f"Total probes: {total}")
        for k in rank_k:
            print(f"Rank-{k} Accuracy: {accuracy[k]:.2f}%")

    # Generate all charts
    print("\n" + "="*50)
    print("Generating evaluation reports...")
    print("="*50)
    
    plot_rank_accuracy(results, rank_k)
    print(f"✓ Saved: rank_accuracy_comparison.png")
    
    plot_cumulative_match_curve(results, rank_k)
    print(f"✓ Saved: cmc_curve.png")
    
    plot_score_distribution(all_scores)
    print(f"✓ Saved: score_distribution.png")
    
    plot_per_attack_stats(results)
    print(f"✓ Saved: attack_statistics.png")
    
    plot_confusion_heatmap(results, rank_k)
    print(f"✓ Saved: rank1_breakdown.png")
    
    generate_summary_report(results, rank_k)
    print(f"✓ Saved: evaluation_summary.txt")
    
    print(f"\nAll reports saved to '{REPORT_DIR}/' directory")


if __name__ == "__main__":
    evaluate_altered()