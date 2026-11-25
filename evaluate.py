"""Final test-set evaluation of one trained model (Part C).

    python evaluate.py --run runs/deep_aug_set1_seed42

Run this ONCE per finished model (data_manual rule 2: the test set is for final scores
only). It refuses to run twice unless you add --force.

Output in the run folder:
    test_results.json      macro-F1 per condition (clean, snr20, snr10, snr0) and overall,
                           per-class F1, confusion matrices
    test_predictions.csv   one row per test clip: true label, prediction, confidence,
                           condition, place, SNR (for Temur's error analysis)
    confusion_matrix.png   overall + clean + 0 dB confusion matrices
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

import data
from model import build_model

SHORT = ["engine", "air_cond", "dog_bark", "children", "street_mus"]   # labels for the plots


def load_checkpoint(run_dir, device="cpu"):
    ckpt = torch.load(Path(run_dir) / "best.pth", map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], n_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.inference_mode()
def predict_probs(model, X, device, batch_size=128):
    probs = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size].to(device))
        probs.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(probs).numpy()


def scores(y_true, y_pred):
    labels = list(range(len(data.CLASSES)))
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class_f1": dict(zip(data.CLASSES, f1_score(y_true, y_pred, average=None, labels=labels,
                                                         zero_division=0).round(4).tolist())),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusions(results, out_path, title):
    panels = [("overall", "All test clips"), ("clean", "Clean"), ("snr0", "0 dB (noisiest)")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (key, label) in zip(axes, panels):
        cm = np.array(results[key]["confusion_matrix"])
        row_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)   # each row sums to 100%
        ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if row_pct[i, j] > 0.5 else "black", fontsize=10)
        ax.set_xticks(range(len(SHORT))); ax.set_xticklabels(SHORT, rotation=35, ha="right")
        ax.set_yticks(range(len(SHORT))); ax.set_yticklabels(SHORT)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{label}  (macro-F1 {results[key]['macro_f1']:.3f})")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained model on the test set")
    p.add_argument("--run", required=True, help="run folder, e.g. runs/deep_aug_set1_seed42")
    p.add_argument("--data-dir", default=str(data.DEFAULT_DATA_DIR))
    p.add_argument("--cache-dir", default=str(data.DEFAULT_CACHE_DIR))
    p.add_argument("--force", action="store_true", help="evaluate again even if already done")
    args = p.parse_args()

    run_dir = Path(args.run)
    if (run_dir / "test_results.json").exists() and not args.force:
        raise SystemExit(f"{run_dir} was already evaluated on the test set. "
                         "The test set is used once per model; add --force only to fix a bug.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(run_dir, device)
    print(f"Model {ckpt['model_name']} | trained on trainSet{ckpt['train_set']} | "
          f"stats {ckpt['stats_file']} | best epoch {ckpt['epoch']}")

    X, y, rows = data.load_list("testSet", args.data_dir, args.cache_dir)
    X = data.scale(X, ckpt["mean"], ckpt["std"])        # SAME stats as training
    probs = predict_probs(model, X, device)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    cond = np.array([r["condition"] for r in rows])

    results = {"run": run_dir.name, "stats_file": ckpt["stats_file"], "overall": scores(y, pred)}
    for c in data.TEST_CONDITIONS:
        mask = cond == c
        results[c] = scores(y[mask], pred[mask])

    with open(run_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(run_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "condition", "place", "snr_db", "true", "pred", "confidence", "correct"])
        for r, t, pr, c in zip(rows, y, pred, conf):
            w.writerow([r["file"], r["condition"], r.get("place", ""), r.get("snr_db", ""),
                        data.CLASSES[t], data.CLASSES[pr], f"{c:.4f}", int(t == pr)])

    plot_confusions(results, run_dir / "confusion_matrix.png", run_dir.name)

    print(f"\n{'condition':10s} {'n':>5s} {'macro-F1':>9s} {'accuracy':>9s}")
    for c in data.TEST_CONDITIONS + ["overall"]:
        r = results[c]
        print(f"{c:10s} {r['n']:5d} {r['macro_f1']:9.4f} {r['accuracy']:9.4f}")
    print(f"\nSaved test_results.json, test_predictions.csv and confusion_matrix.png in {run_dir}")


if __name__ == "__main__":
    main()
