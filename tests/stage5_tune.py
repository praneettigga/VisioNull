#!/usr/bin/env python3
"""
Stage 5: Threshold Tuning & Model Improvement
Sweeps fall detection thresholds on the dataset to find optimal values.
Optionally trains a simple ML classifier on extracted pose features
and compares its accuracy against the rule-based detector.

Usage:
    python3 tests/stage5_tune.py
    python3 tests/stage5_tune.py --dataset-path data/cctv-incident/images
    python3 tests/stage5_tune.py --ml   # Also train ML classifier
"""

import sys
import os
import argparse
import csv
import time
import itertools

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.pose_estimator import PoseEstimator
from src.fall_detector import FallDetector, FallState
from src.dataset_stream import DatasetStream
from typing import Optional, List, Dict, Tuple


# Ground truth mapping
GT_MAP = {"laying": True, "standing": False}


def extract_features_from_dataset(
    source_path: str,
    labels_path: Optional[str] = None
) -> Tuple[List[dict], List[bool], List[str]]:
    """
    Run pose estimation on the dataset and extract features + ground truth labels.

    Returns:
        features: List of feature dicts per image (body_angle, head_y, etc.)
        labels: List of bool (True = fall, False = not fall)
        image_names: List of image filenames
    """
    pose = PoseEstimator(model_complexity=0)
    ds = DatasetStream(source_path, labels_path=labels_path)

    if not ds.start():
        print("  Failed to start DatasetStream")
        pose.close()
        return [], [], []

    features = []
    labels = []
    image_names = []

    frame_idx = 0
    while True:
        frame = ds.get_frame()
        if frame is None:
            break
        frame_idx += 1

        gt = ds.get_ground_truth()
        h, w = frame.shape[:2]
        landmarks = pose.get_landmarks(frame)

        if landmarks is None or gt is None:
            continue

        # Extract features using a fresh detector (to get metrics)
        detector = FallDetector()
        state, metrics = detector.update(landmarks, h)

        for ann in gt:
            is_fall = GT_MAP.get(ann.class_name)
            if is_fall is None:
                continue

            feat = {
                "body_angle": metrics.body_angle,
                "head_height_ratio": metrics.head_height_ratio,
                "head_velocity": metrics.head_velocity,
                "shoulder_hip_ratio": metrics.shoulder_hip_ratio,
                "hip_height_ratio": metrics.hip_height_ratio,
                "leg_compression": metrics.leg_compression,
            }

            features.append(feat)
            labels.append(is_fall)
            img_path = ds.get_current_image_path()
            image_names.append(os.path.basename(img_path) if img_path else f"frame_{frame_idx}")

        if frame_idx % 20 == 0:
            print(f"    Extracted features from {frame_idx} frames...")

    ds.stop()
    pose.close()

    print(f"  Extracted {len(features)} samples ({sum(labels)} fall, {len(labels) - sum(labels)} standing)")
    return features, labels, image_names


def evaluate_thresholds(
    features: List[dict],
    labels: List[bool],
    fall_head_threshold: float,
    horizontal_ratio_threshold: float,
    hip_height_threshold: float = 0.65,
    leg_compression_threshold: float = 0.6,
) -> dict:
    """Evaluate a threshold configuration against the dataset features."""
    y_true = labels
    y_pred = []

    for feat in features:
        is_horizontal = (
            feat["shoulder_hip_ratio"] > horizontal_ratio_threshold or
            feat["body_angle"] > 45
        )
        is_head_low = feat["head_height_ratio"] > fall_head_threshold
        is_on_ground = (
            feat["hip_height_ratio"] > hip_height_threshold or
            feat["leg_compression"] > leg_compression_threshold
        )

        is_fall = (
            (is_horizontal and is_head_low) or
            (is_on_ground and feat["head_height_ratio"] > 0.40)
        )
        y_pred.append(is_fall)

    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
    }


def sweep_thresholds(features: List[dict], labels: List[bool]) -> List[dict]:
    """Sweep threshold combinations and find the best ones."""
    print("\n── Threshold Sweep ──")

    # Parameter ranges to sweep
    head_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    horizontal_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    hip_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    leg_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = []
    total = len(head_thresholds) * len(horizontal_thresholds) * len(hip_thresholds) * len(leg_thresholds)
    print(f"  Testing {total} combinations...")

    for ht, hrt, hipt, lct in itertools.product(
        head_thresholds, horizontal_thresholds, hip_thresholds, leg_thresholds
    ):
        m = evaluate_thresholds(features, labels, ht, hrt, hipt, lct)
        m["head_threshold"] = ht
        m["horizontal_threshold"] = hrt
        m["hip_threshold"] = hipt
        m["leg_threshold"] = lct
        results.append(m)

    # Sort by F1 score
    results.sort(key=lambda x: x["f1"], reverse=True)

    # Show top 10
    print(f"\n  Top 10 Configurations (by F1 Score):")
    print(f"  {'Head':>6s} {'Horiz':>6s} {'Hip':>6s} {'Leg':>6s}  "
          f"{'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'Acc':>6s}  "
          f"{'TP':>3s} {'FP':>3s} {'FN':>3s} {'TN':>3s}")
    print(f"  {'─'*72}")

    for r in results[:10]:
        print(f"  {r['head_threshold']:>6.2f} {r['horizontal_threshold']:>6.2f}"
              f" {r['hip_threshold']:>6.2f} {r['leg_threshold']:>6.2f}  "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['accuracy']:>6.3f}  "
              f"{r['tp']:>3d} {r['fp']:>3d} {r['fn']:>3d} {r['tn']:>3d}")

    # Current defaults comparison
    print(f"\n  Current defaults:")
    default = evaluate_thresholds(features, labels, 0.55, 0.6, 0.65, 0.6)
    print(f"  Head=0.55, Horiz=0.60, Hip=0.65, Leg=0.60 →"
          f" Prec={default['precision']:.3f}, Recall={default['recall']:.3f},"
          f" F1={default['f1']:.3f}, Acc={default['accuracy']:.3f}")

    best = results[0]
    print(f"\n  Best found:")
    print(f"  Head={best['head_threshold']:.2f}, Horiz={best['horizontal_threshold']:.2f},"
          f" Hip={best['hip_threshold']:.2f}, Leg={best['leg_threshold']:.2f} →"
          f" Prec={best['precision']:.3f}, Recall={best['recall']:.3f},"
          f" F1={best['f1']:.3f}, Acc={best['accuracy']:.3f}")

    return results


def train_ml_classifier(features: List[dict], labels: List[bool]):
    """Train a simple ML classifier on pose features and compare."""
    print("\n── ML Classifier Comparison ──")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  scikit-learn not installed. Run: pip install scikit-learn")
        return

    # Prepare feature matrix
    feature_names = ["body_angle", "head_height_ratio", "head_velocity",
                     "shoulder_hip_ratio", "hip_height_ratio", "leg_compression"]
    X = np.array([[f[k] for k in feature_names] for f in features])
    y = np.array(labels, dtype=int)

    if len(np.unique(y)) < 2:
        print("  Only one class in the dataset — cannot train classifier")
        return

    print(f"  Samples: {len(X)} ({sum(y)} fall, {len(y) - sum(y)} standing)")
    print(f"  Features: {feature_names}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    cv = StratifiedKFold(n_splits=min(5, min(sum(y), len(y) - sum(y))), shuffle=True, random_state=42)

    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
    }

    for name, clf in classifiers.items():
        try:
            scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1')
            print(f"\n  {name}:")
            print(f"    F1 (cross-val): {scores.mean():.3f} ± {scores.std():.3f}")
            print(f"    Per-fold: {[f'{s:.3f}' for s in scores]}")

            # Fit on full data and show feature importances
            clf.fit(X_scaled, y)
            if hasattr(clf, 'feature_importances_'):
                importances = sorted(zip(feature_names, clf.feature_importances_),
                                   key=lambda x: x[1], reverse=True)
                print(f"    Feature importance:")
                for fn, imp in importances:
                    print(f"      {fn:<25s} {imp:.3f}")
            elif hasattr(clf, 'coef_'):
                coefs = sorted(zip(feature_names, np.abs(clf.coef_[0])),
                             key=lambda x: x[1], reverse=True)
                print(f"    Feature importance (|coef|):")
                for fn, c in coefs:
                    print(f"      {fn:<25s} {c:.3f}")

        except Exception as e:
            print(f"  {name}: Error — {e}")


def save_results(results: List[dict], output_path: str):
    """Save sweep results to CSV."""
    fieldnames = ["head_threshold", "horizontal_threshold", "hip_threshold", "leg_threshold",
                  "precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\n  Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 5: Threshold Tuning")
    parser.add_argument("--dataset-path", type=str, help="Path to dataset images")
    parser.add_argument("--labels", type=str, help="Path to labels directory")
    parser.add_argument("--ml", action="store_true", help="Also train ML classifier")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    print("=" * 50)
    print("  VisioNull — Stage 5: Threshold Tuning")
    print("=" * 50)

    # Find dataset
    source = args.dataset_path
    if source is None:
        candidates = [
            os.path.join(PROJECT_ROOT, "data", "cctv-incident", "images"),
            os.path.join(PROJECT_ROOT, "data", "cctv-incident", "train", "images"),
            os.path.join(PROJECT_ROOT, "data", "cctv-incident"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                source = c
                break
        if source is None:
            print("\n  Dataset not found. Run: bash tests/download_dataset.sh")
            return 1

    # Step 1: Extract features
    print("\n── Extracting Pose Features ──")
    features, labels, image_names = extract_features_from_dataset(source, labels_path=args.labels)

    if not features:
        print("  No features extracted. Check dataset.")
        return 1

    # Step 2: Sweep thresholds
    results = sweep_thresholds(features, labels)

    # Step 3: Save results
    output_path = args.output or os.path.join(PROJECT_ROOT, "tests", "tuning_results.csv")
    save_results(results, output_path)

    # Step 4: ML comparison (optional)
    if args.ml:
        train_ml_classifier(features, labels)

    print(f"\n{'=' * 50}")
    print("  Stage 5 complete")
    print("  Review tuning_results.csv and update src/config.py")
    print("  Proceed to Stage 6 (full pipeline)")
    print(f"{'=' * 50}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
