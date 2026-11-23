"""Use a trained model: the functions the app imports, plus the "not sure" threshold (Part C).

Choose the threshold on the VALIDATION set (never the test set), once per final model:
    python predict.py --run runs/deep_aug_set2_seed42 --tune-threshold

Classify a 2-second, 16 kHz WAV (for a quick check):
    python predict.py --run runs/deep_aug_set2_seed42 --file ../data/original/test/dog_bark/xxx.wav

In the app (Part E):
    from predict import load_bundle, predict_scaled, decide
    bundle = load_bundle("runs/deep_aug_set2_seed42")     # model + stats + classes + threshold
    probs = predict_scaled(bundle, windows)               # windows from adapter_for_app, already
                                                          # scaled with bundle["stats_file"]
    label, confidence = decide(bundle, probs.mean(axis=0))   # average the windows of one file
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import data
import features as F
from evaluate import load_checkpoint, predict_probs

DEFAULT_THRESHOLD = 0.5
NOT_SURE = "not sure"


def load_bundle(run_dir, device="cpu"):
    """Everything needed to make predictions with one trained model."""
    run_dir = Path(run_dir)
    model, ckpt = load_checkpoint(run_dir, device)
    threshold = DEFAULT_THRESHOLD
    if (run_dir / "threshold.json").exists():
        threshold = json.loads((run_dir / "threshold.json").read_text())["threshold"]
    return {"model": model, "device": device, "classes": ckpt["classes"],
            "mean": ckpt["mean"], "std": ckpt["std"], "stats_file": ckpt["stats_file"],
            "threshold": threshold}


def _as_batch(x):
    x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    if x.dim() == 2:          # (64, 201) -> (1, 1, 64, 201)
        x = x[None, None]
    elif x.dim() == 3:        # (N, 64, 201) -> (N, 1, 64, 201)
        x = x[:, None]
    return x


def predict_scaled(bundle, x):
    """Class probabilities (N, 5) for spectrograms that are ALREADY scaled
    with this model's stats file (e.g. the output of adapter_for_app)."""
    return predict_probs(bundle["model"], _as_batch(x), bundle["device"])


def predict_logmel(bundle, x):
    """Class probabilities (N, 5) for UNSCALED log-mels from features.log_mel."""
    x = (np.asarray(x, dtype=np.float32) - bundle["mean"]) / bundle["std"]
    return predict_scaled(bundle, x)


def decide(bundle, probs):
    """One probability vector -> (label or "not sure", confidence)."""
    probs = np.asarray(probs)
    idx = int(probs.argmax())
    conf = float(probs[idx])
    if conf < bundle["threshold"]:
        return NOT_SURE, conf
    return bundle["classes"][idx], conf


def predict_file(bundle, wav_path):
    """Classify one 2 s clip in the dataset format (for checks; the app uses adapter_for_app)."""
    probs = predict_logmel(bundle, F.log_mel(F.load_clip(wav_path)))[0]
    return decide(bundle, probs), probs


def tune_threshold(run_dir, target_accuracy=0.90, data_dir=data.DEFAULT_DATA_DIR,
                   cache_dir=data.DEFAULT_CACHE_DIR):
    """Pick the lowest confidence threshold at which the answers the model DOES give
    are at least target_accuracy correct on the validation set. Below it, the app says "not sure"."""
    bundle = load_bundle(run_dir)
    X, y, _ = data.load_list("valSet", data_dir, cache_dir)
    probs = predict_logmel(bundle, X[:, 0])
    pred, conf = probs.argmax(axis=1), probs.max(axis=1)

    table = []
    for t in np.round(np.arange(0.30, 0.96, 0.05), 2):
        answered = conf >= t
        coverage = float(answered.mean())
        acc = float((pred[answered] == y[answered]).mean()) if answered.any() else float("nan")
        table.append({"threshold": float(t), "answered": coverage, "accuracy_when_answered": acc})

    chosen = next((r for r in table if r["accuracy_when_answered"] >= target_accuracy), table[-1])
    result = {"threshold": chosen["threshold"], "target_accuracy": target_accuracy,
              "val_answered": chosen["answered"],
              "val_accuracy_when_answered": chosen["accuracy_when_answered"],
              "val_accuracy_no_threshold": float((pred == y).mean()), "table": table}
    (Path(run_dir) / "threshold.json").write_text(json.dumps(result, indent=2))

    print(f"{'threshold':>9s} {'answered':>9s} {'acc. when answered':>19s}")
    for r in table:
        mark = "  <- chosen" if r is chosen else ""
        acc = r["accuracy_when_answered"]
        acc_txt = f"{acc:19.1%}" if acc == acc else f"{'(none answered)':>19s}"   # acc != acc means NaN
        print(f"{r['threshold']:9.2f} {r['answered']:9.1%} {acc_txt}{mark}")
    print(f"\nThreshold {chosen['threshold']:.2f}: the model answers {chosen['answered']:.0%} of "
          f"validation clips and is right {chosen['accuracy_when_answered']:.0%} of the time; "
          f"the rest get 'not sure'. Saved to threshold.json")
    return result


def main():
    p = argparse.ArgumentParser(description="Predict with a trained SoundSense model")
    p.add_argument("--run", required=True)
    p.add_argument("--tune-threshold", action="store_true")
    p.add_argument("--target-accuracy", type=float, default=0.90)
    p.add_argument("--file", help="a 2 s, 16 kHz WAV clip to classify")
    p.add_argument("--data-dir", default=str(data.DEFAULT_DATA_DIR))
    p.add_argument("--cache-dir", default=str(data.DEFAULT_CACHE_DIR))
    args = p.parse_args()

    if args.tune_threshold:
        tune_threshold(args.run, args.target_accuracy, args.data_dir, args.cache_dir)
    if args.file:
        bundle = load_bundle(args.run)
        (label, conf), probs = predict_file(bundle, args.file)
        print(f"Prediction: {label} ({conf:.0%})  [threshold {bundle['threshold']:.2f}]")
        for c, pr in sorted(zip(bundle["classes"], probs), key=lambda t: -t[1]):
            print(f"  {c:17s} {pr:.1%}")
    if not (args.tune_threshold or args.file):
        p.print_help()


if __name__ == "__main__":
    main()
