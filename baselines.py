"""Baselines for the experiment log .

    python baselines.py

Two baselines, each trained on trainSet1, trainSet2 and trainSet3 (data_manual: "Baselines first"):

  majority   always predicts the most common class of the training set.
  mfcc_lr    20 MFCCs per frame (DCT of the team's features.log_mel, so the preprocessing is the
             same one the CNNs use) -> mean and std over time (40 numbers per clip) -> standardized
             -> multinomial logistic regression. C is chosen on valSet.csv only.

Every baseline is scored on testSet.csv exactly once, per condition (clean, snr20, snr10, snr0) and
overall, with macro-F1. Output goes to runs/baseline_<name>_set<k>/ in the same format as
evaluate.py (summary.json, test_results.json, test_predictions.csv, confusion_matrix.png), so
summarize.py puts the baselines into the same results table as the CNNs.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.fft import dct
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data
from evaluate import plot_confusions, scores

REPO = Path(__file__).resolve().parent
N_MFCC = 20
C_GRID = [0.01, 0.1, 1.0, 10.0]
SEED = 42


def mfcc_stats(X):
    """(N, 1, 64, 201) unscaled log-mels (dB) -> (N, 40): mean and std over time of 20 MFCCs."""
    mfcc = dct(X[:, 0], type=2, axis=1, norm="ortho")[:, :N_MFCC, :]   # (N, 20, 201)
    return np.concatenate([mfcc.mean(axis=2), mfcc.std(axis=2)], axis=1)


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro", labels=list(range(len(data.CLASSES))),
                          zero_division=0))


def save_run(out_dir, summary, y_test, pred, conf, rows):
    """Write the same files evaluate.py writes for a CNN."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cond = np.array([r["condition"] for r in rows])
    results = {"run": out_dir.name, "stats_file": None, "overall": scores(y_test, pred)}
    for c in data.TEST_CONDITIONS:
        m = cond == c
        results[c] = scores(y_test[m], pred[m])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "test_results.json").write_text(json.dumps(results, indent=2))
    with open(out_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "condition", "place", "snr_db", "true", "pred", "confidence", "correct"])
        for r, t, p, c in zip(rows, y_test, pred, conf):
            w.writerow([r["file"], r["condition"], r.get("place", ""), r.get("snr_db", ""),
                        data.CLASSES[t], data.CLASSES[p], f"{c:.4f}", int(t == p)])
    plot_confusions(results, out_dir / "confusion_matrix.png", out_dir.name)
    return results


def main():
    p = argparse.ArgumentParser(description="Majority-class and MFCC + logistic regression baselines")
    p.add_argument("--data-dir", default=str(data.DEFAULT_DATA_DIR))
    p.add_argument("--cache-dir", default=str(data.DEFAULT_CACHE_DIR))
    p.add_argument("--runs-dir", default=str(REPO / "runs"))
    args = p.parse_args()
    runs = Path(args.runs_dir)

    X_val, y_val, _ = data.load_list("valSet", args.data_dir, args.cache_dir)
    X_test, y_test, test_rows = data.load_list("testSet", args.data_dir, args.cache_dir)
    F_val, F_test = mfcc_stats(X_val), mfcc_stats(X_test)

    print(f"{'run':28s} {'val':>6s} {'clean':>6s} {'snr20':>6s} {'snr10':>6s} {'snr0':>6s} {'all':>6s}")
    for k in (1, 2, 3):
        X_tr, y_tr, _ = data.load_list(f"trainSet{k}", args.data_dir, args.cache_dir)
        base = {"augment": False, "train_set": k, "seed": SEED, "limit": 0, "best_epoch": "-"}

        # ---- majority class (ties -> lowest class id, i.e. deterministic)
        maj = int(np.bincount(y_tr, minlength=len(data.CLASSES)).argmax())
        pred = np.full(len(y_test), maj)
        summary = {"run": f"baseline_majority_set{k}", "model": "majority", **base,
                   "best_val_macro_f1": macro_f1(y_val, np.full(len(y_val), maj)),
                   "majority_class": data.CLASSES[maj]}
        res = save_run(runs / summary["run"], summary, y_test, pred, np.ones(len(y_test)), test_rows)
        report(summary, res)

        # ---- MFCC + logistic regression, C picked on validation
        F_tr = mfcc_stats(X_tr)
        best = None
        for C in C_GRID:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=C, max_iter=5000, random_state=SEED))
            clf.fit(F_tr, y_tr)
            f1 = macro_f1(y_val, clf.predict(F_val))
            if best is None or f1 > best[0]:
                best = (f1, C, clf)
        val_f1, C, clf = best
        probs = clf.predict_proba(F_test)
        summary = {"run": f"baseline_mfcc_lr_set{k}", "model": "mfcc_lr", **base,
                   "best_val_macro_f1": val_f1, "C": C, "C_grid": C_GRID,
                   "features": f"{N_MFCC} MFCC (DCT of features.log_mel) mean+std over time"}
        res = save_run(runs / summary["run"], summary, y_test, probs.argmax(1), probs.max(1), test_rows)
        report(summary, res)

    print("\nSaved runs/baseline_*/. Run summarize.py to update runs/results.md.")


def report(s, r):
    cols = [r[c]["macro_f1"] for c in data.TEST_CONDITIONS + ["overall"]]
    print(f"{s['run']:28s} {s['best_val_macro_f1']:6.3f} " + " ".join(f"{v:6.3f}" for v in cols))


if __name__ == "__main__":
    main()
