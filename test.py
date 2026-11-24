"""Evaluate a fine-tuned multi-disease model on colour fundus photographs.

Predictions are made per image, averaged over the available eyes of each
participant, and evaluated at participant level.

For each disease the model outputs two probabilities: prior (present at
baseline) and incident (present at baseline or during follow-up).

Example (decision thresholds tuned on the validation set):
    python test.py --checkpoint best_model.pth --test_csv test.csv --val_csv val.csv \
        --image_dir images/ --output_dir results/
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import FundusDiseaseDataset
from model import load_model

TASKS = ("prior", "incident")


def predict(model, csv_file, image_dir, disease_names, args, device):
    """Return participant IDs, eye-averaged probabilities [N, D, 2] and labels [N, D, 2]."""
    dataset = FundusDiseaseDataset(csv_file, image_dir, disease_names)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    probs, labels, ids = [], [], []
    for images, batch_labels, batch_ids in loader:
        probs.append(model.predict_proba(images.to(device)).cpu().numpy())
        labels.append(batch_labels.numpy())
        ids.extend(batch_ids)
    probs, labels = np.concatenate(probs), np.concatenate(labels)

    # average the left- and right-eye predictions of each participant
    ids, first, inverse = np.unique(np.array(ids), return_index=True, return_inverse=True)
    sums = np.zeros((len(ids),) + probs.shape[1:])
    np.add.at(sums, inverse, probs)
    counts = np.bincount(inverse).reshape(-1, 1, 1)
    return ids, sums / counts, labels[first]


def auc_or_nan(y_true, y_score):
    return roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan


def best_f1_threshold(y_true, y_prob):
    """Threshold that maximises F1 (used on the validation set only)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_threshold, best_f1 = 0.5, 0.0
    for threshold in np.linspace(0, 1, 100):
        f1 = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return best_threshold


def evaluate(probs, labels, disease_names, thresholds=None):
    rows = []
    for d, name in enumerate(disease_names):
        for t, task in enumerate(TASKS):
            valid = ~np.isnan(labels[:, d, t])
            if not valid.any():
                continue
            y_true, y_prob = labels[valid, d, t].astype(int), probs[valid, d, t]
            threshold = thresholds[(name, task)] if thresholds else 0.5
            y_pred = (y_prob >= threshold).astype(int)
            rows.append({
                "Disease": name, "Task": task, "N": int(valid.sum()),
                "AUC": auc_or_nan(y_true, y_prob),
                "Acc": accuracy_score(y_true, y_pred),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "Threshold": threshold,
            })
    metrics = pd.DataFrame(rows)
    if len(metrics):
        for task in TASKS + ("overall",):
            sub = metrics if task == "overall" else metrics[metrics["Task"] == task]
            sub = sub[sub["AUC"].notna() & (sub["Disease"] != "Average")]
            metrics.loc[len(metrics)] = {"Disease": "Average", "Task": task,
                                         "AUC": sub["AUC"].mean(), "F1": sub["F1"].mean()}
        metrics["N"] = metrics["N"].astype("Int64")
    return metrics


def save_predictions(path, ids, probs, disease_names):
    columns = {"sno": ids}
    for d, disease in enumerate(disease_names):
        for t, task in enumerate(TASKS):
            columns[f"pred_{task}_{disease}"] = probs[:, d, t]
    pd.DataFrame(columns).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="fine-tuned model weights (.pth)")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--val_csv", default=None,
                        help="tune per-disease F1 thresholds on this set (default: fixed 0.5)")
    parser.add_argument("--val_image_dir", default=None, help="defaults to --image_dir")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    model, disease_names = load_model(args.checkpoint, args.hidden_dim, device)

    thresholds = None
    if args.val_csv:
        _, val_probs, val_labels = predict(model, args.val_csv, args.val_image_dir or args.image_dir,
                                           disease_names, args, device)
        thresholds = {}
        for d, name in enumerate(disease_names):
            for t, task in enumerate(TASKS):
                valid = ~np.isnan(val_labels[:, d, t])
                thresholds[(name, task)] = best_f1_threshold(val_labels[valid, d, t].astype(int),
                                                             val_probs[valid, d, t])

    ids, probs, labels = predict(model, args.test_csv, args.image_dir, disease_names, args, device)
    save_predictions(os.path.join(args.output_dir, "predictions.csv"), ids, probs, disease_names)

    metrics = evaluate(probs, labels, disease_names, thresholds)

    if len(metrics):
        metrics.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False, float_format="%.4f")
        print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    else:
        print("No label columns found: predictions saved, metrics skipped.")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
