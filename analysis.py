"""Model choice, significance checks and error analysis for the report (Part D).

    python analysis.py

Reads only files that already exist: runs/*/summary.json, test_results.json, test_predictions.csv,
runs/<final>/threshold.json and data/testSet.csv. It never runs a model, so it does not touch the
test set again (data_manual rule 2).

Output in runs/analysis/:
    analysis.json            every number quoted in RESULTS.md
    snr_curves.png           macro-F1 vs noise level for the key comparison + baselines
    per_class_f1.png         per-class F1 of the final model at each noise level
    errors_by_place.png      error rate of the final model per background place
    confident_errors.csv     the final model's most confident mistakes (for blind listening)
    flipped_by_noise.csv     test events the final model gets right clean but wrong at 0 dB
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent
RUNS = REPO / "runs"
OUT = RUNS / "analysis"
CLASSES = ["engine_idling", "air_conditioner", "dog_bark", "children_playing", "street_music"]
CONDS = ["clean", "snr20", "snr10", "snr0"]
COND_LABEL = {"clean": "Clean", "snr20": "20 dB", "snr10": "10 dB", "snr0": "0 dB"}
N_BOOT = 2000
SEED = 0

# Reference categorical palette (dataviz skill, slots 1-4 in fixed order) + recessive inks
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_preds(run):
    with open(RUNS / run / "test_predictions.csv", newline="") as f:
        return list(csv.DictReader(f))


def macro_f1(rows):
    y = [CLASSES.index(r["true"]) for r in rows]
    p = [CLASSES.index(r["pred"]) for r in rows]
    return f1_score(y, p, average="macro", labels=list(range(5)), zero_division=0)


def event_key(test_row):
    """The 4 versions of one test event (clean, 20, 10, 0 dB) share a key, so the bootstrap
    resamples EVENTS, not rows (data_manual: the 1,993 rows are not independent)."""
    return test_row["source_file"]


def paired_bootstrap(run_a, run_b, test_rows, conds=None):
    """Difference in macro-F1 (a - b) with a 95% interval, resampling test events."""
    a, b = load_preds(run_a), load_preds(run_b)
    assert [r["file"] for r in a] == [r["file"] for r in b] == [r["file"] for r in test_rows]
    idx = [i for i, r in enumerate(test_rows) if conds is None or r["condition"] in conds]
    groups = defaultdict(list)
    for i in idx:
        groups[event_key(test_rows[i])].append(i)
    keys = list(groups)
    ya = np.array([CLASSES.index(r["true"]) for r in a])
    pa = np.array([CLASSES.index(r["pred"]) for r in a])
    pb = np.array([CLASSES.index(r["pred"]) for r in b])
    f = lambda y, p: f1_score(y, p, average="macro", labels=list(range(5)), zero_division=0)
    point = f(ya[idx], pa[idx]) - f(ya[idx], pb[idx])
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(N_BOOT):
        sel = np.concatenate([groups[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        diffs.append(f(ya[sel], pa[sel]) - f(ya[sel], pb[sel]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"a": run_a, "b": run_b, "conditions": conds or "all", "diff": round(float(point), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "significant": bool(lo > 0 or hi < 0), "n_events": len(keys)}


def main():
    OUT.mkdir(exist_ok=True)
    with open(REPO / "data" / "testSet.csv", newline="") as f:
        test_rows = list(csv.DictReader(f))

    runs = {}
    for s in sorted(RUNS.glob("*/summary.json")):
        summ = json.loads(s.read_text())
        t = s.parent / "test_results.json"
        if t.exists() and not summ.get("limit"):
            runs[summ["run"]] = {"summary": summ, "test": json.loads(t.read_text())}

    # ---------------------------------------------------------------- 1. final model
    # Chosen on VALIDATION macro-F1 only, among the CNNs (the app needs a CNN checkpoint).
    cnns = {k: v for k, v in runs.items() if not k.startswith("baseline")}
    final = max(cnns, key=lambda k: cnns[k]["summary"]["best_val_macro_f1"])
    val_rank = sorted(cnns, key=lambda k: -cnns[k]["summary"]["best_val_macro_f1"])

    # ---------------------------------------------------------------- 2. significance
    comparisons = [
        ("deep_aug_set3_seed42", "deep_aug_set1_seed42", None),
        ("deep_aug_set3_seed42", "deep_aug_set2_seed42", None),
        ("deep_aug_set2_seed42", "deep_aug_set1_seed42", None),
        ("deep_aug_set3_seed42", "deep_aug_set1_seed42", ["clean"]),
        ("deep_aug_set3_seed42", "deep_aug_set2_seed42", ["clean"]),
        ("deep_aug_set3_seed42", "deep_aug_set1_seed42", ["snr0"]),
        ("deep_aug_set3_seed42", "deep_aug_set2_seed42", ["snr0"]),
        ("deep_aug_set2_seed42", "deep_set2_seed42", None),          # augmentation
        ("deep_set2_seed42", "small_set2_seed42", None),             # depth + batch norm
        ("deep_aug_set3_seed42", "baseline_mfcc_lr_set3", None),     # CNN vs baseline
        ("deep_aug_set1_seed42", "baseline_mfcc_lr_set1", None),
    ]
    sig = [paired_bootstrap(a, b, test_rows, c) for a, b, c in comparisons]

    # ---------------------------------------------------------------- 3. noise sanity check
    sanity = {}
    for k, v in runs.items():
        if "majority" in k:
            continue
        f1s = [v["test"][c]["macro_f1"] for c in CONDS]
        sanity[k] = {"clean_minus_snr0": round(f1s[0] - f1s[3], 4),
                     "snr20_to_snr0_monotonic": bool(f1s[1] >= f1s[2] >= f1s[3])}

    # ---------------------------------------------------------------- 4. error analysis (final)
    preds = load_preds(final)
    for p, t in zip(preds, test_rows):
        p["source_file"] = t["source_file"]
    errors = [p for p in preds if p["correct"] == "0"]
    pairs = Counter((p["true"], p["pred"]) for p in errors)
    top_pairs = [{"true": t, "pred": q, "n": n, "share_of_errors": round(n / len(errors), 3)}
                 for (t, q), n in pairs.most_common(8)]

    per_cond_pairs = {}
    for c in CONDS:
        e = [p for p in errors if p["condition"] == c]
        per_cond_pairs[c] = {"errors": len(e), "of": sum(p["condition"] == c for p in preds),
                             "top": [f"{t} -> {q}: {n}" for (t, q), n in
                                     Counter((p["true"], p["pred"]) for p in e).most_common(3)]}

    by_place = {}
    mixed = [p for p in preds if p["condition"] != "clean"]
    for place in sorted({p["place"] for p in mixed}):
        rows = [p for p in mixed if p["place"] == place]
        e = [p for p in rows if p["correct"] == "0"]
        by_place[place] = {
            "n": len(rows), "error_rate": round(len(e) / len(rows), 4),
            "macro_f1": round(macro_f1(rows), 4),
            "pred_air_conditioner_when_wrong": sum(p["pred"] == "air_conditioner" for p in e),
            "errors": len(e),
            "error_rate_by_snr": {c: round(np.mean([p["correct"] == "0" for p in rows
                                                    if p["condition"] == c]), 4) for c in CONDS[1:]}}

    # The same event at clean / 20 / 10 / 0 dB: how many correct clean answers does noise break?
    # (clean originals and their mixes share the source_file; engine_idling repeats are rare)
    by_event = defaultdict(dict)
    for p in preds:
        by_event[p["source_file"]][p["condition"]] = p
    flipped = [(ev, d) for ev, d in by_event.items()
               if "clean" in d and "snr0" in d and d["clean"]["correct"] == "1" and d["snr0"]["correct"] == "0"]
    with open(OUT / "flipped_by_noise.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_file", "true", "pred_clean", "pred_snr20", "pred_snr10", "pred_snr0", "place", "snr0_file"])
        for ev, d in sorted(flipped, key=lambda x: x[1]["clean"]["true"]):
            w.writerow([ev, d["clean"]["true"], d["clean"]["pred"], d.get("snr20", {}).get("pred", ""),
                        d.get("snr10", {}).get("pred", ""), d["snr0"]["pred"], d["snr0"]["place"], d["snr0"]["file"]])
    flip_pairs = Counter((d["clean"]["true"], d["snr0"]["pred"]) for _, d in flipped).most_common(5)
    paired_events = sum("clean" in d and "snr0" in d for d in by_event.values())

    conf = np.array([float(p["confidence"]) for p in preds])
    ok = np.array([p["correct"] == "1" for p in preds])
    confident = sorted(errors, key=lambda p: -float(p["confidence"]))[:25]
    with open(OUT / "confident_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "condition", "place", "snr_db", "true", "pred", "confidence"])
        w.writeheader()
        for p in confident:
            w.writerow({k: p[k] for k in w.fieldnames})

    # salience: UrbanSound8K marks each event foreground (1) or background (2)
    sal = {}
    us8k = REPO.parent / "data" / "original" / "manifest.csv"
    if us8k.exists():
        with open(us8k, newline="") as f:
            mrows = list(csv.DictReader(f))
        skey = next((k for k in mrows[0] if "salience" in k), None)
        fkey = next((k for k in mrows[0] if k in ("source_file", "slice_file_name")), None)
        if skey and fkey:
            s_of = {r[fkey]: r[skey] for r in mrows}
            clean = [p for p in preds if p["condition"] == "clean"]
            for s in sorted({s_of.get(p["source_file"]) for p in clean} - {None}):
                rows = [p for p in clean if s_of.get(p["source_file"]) == s]
                sal[s] = {"n": len(rows), "accuracy": round(np.mean([p["correct"] == "1" for p in rows]), 4)}

    # ---------------------------------------------------------------- 5. "not sure" on test
    thr_file = json.loads((RUNS / final / "threshold.json").read_text()) if (RUNS / final / "threshold.json").exists() else None
    reject = []
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        ans = conf >= t
        reject.append({"threshold": t, "answered": round(float(ans.mean()), 4),
                       "accuracy_when_answered": round(float(ok[ans].mean()), 4),
                       "answered_by_condition": {c: round(float(ans[[p["condition"] == c for p in preds]].mean()), 4) for c in CONDS}})

    result = {
        "final_model": final,
        "selection_rule": "highest validation macro-F1 among CNN checkpoints (test set not used to choose)",
        "validation_ranking": [(k, round(cnns[k]["summary"]["best_val_macro_f1"], 4)) for k in val_rank],
        "significance": sig,
        "noise_sanity": sanity,
        "final_errors": {"n_errors": len(errors), "n_test": len(preds), "top_confusions": top_pairs,
                         "per_condition": per_cond_pairs, "by_place": by_place,
                         "flipped_clean_to_snr0": {"n": len(flipped), "of_events": paired_events,
                                                   "top": [f"{t} -> {q}: {n}" for (t, q), n in flip_pairs]},
                         "confidence": {"mean_when_correct": round(float(conf[ok].mean()), 4),
                                        "mean_when_wrong": round(float(conf[~ok].mean()), 4),
                                        "errors_with_conf_ge_0.9": int(((~ok) & (conf >= 0.9)).sum())},
                         "clean_accuracy_by_salience": sal},
        "threshold_file": {k: thr_file.get(k) for k in ("threshold", "chosen_by", "target_accuracy",
                                                        "val_answered", "val_accuracy_when_answered")}
                          if thr_file else None,
        "not_sure_on_test": reject,
    }
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2))

    # ---------------------------------------------------------------- figures
    x = np.arange(len(CONDS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    groups = [
        ("Key comparison: training data (deep CNN + aug)",
         [("deep_aug_set1_seed42", "Original only (set 1)", BLUE),
          ("deep_aug_set2_seed42", "Half / half (set 2)", ORANGE),
          ("deep_aug_set3_seed42", "Mixed only (set 3)", AQUA)]),
        ("Model steps, all on set 2  (+ MFCC baseline)",
         [("small_set2_seed42", "Small CNN", BLUE),
          ("deep_set2_seed42", "Deep CNN + batch norm", ORANGE),
          ("deep_aug_set2_seed42", "Deep CNN + aug", AQUA),
          ("baseline_mfcc_lr_set2", "MFCC + logistic reg.", YELLOW)]),
    ]
    for ax, (title, series) in zip(axes, groups):
        style(ax)
        for run, label, color in series:
            y = [runs[run]["test"][c]["macro_f1"] for c in CONDS]
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label)
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK2)
        maj = runs["baseline_majority_set2"]["test"]["overall"]["macro_f1"]
        ax.axhline(maj, color=INK2, linewidth=1, linestyle=":")
        ax.text(3.4, maj + 0.015, f"majority class ({maj:.2f})", fontsize=8, color=INK2, ha="right")
        ax.set_xticks(x, [COND_LABEL[c] for c in CONDS])
        ax.set_xlim(-0.2, 3.45)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Test condition (more noise →)", color=INK2)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(0.0, 0.3), labelcolor=INK)
    axes[0].set_ylabel("Macro-F1 (test)", color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "snr_curves.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    w = 0.2
    for i, (c, color) in enumerate(zip(CONDS, [BLUE, ORANGE, AQUA, YELLOW])):
        vals = [runs[final]["test"][c]["per_class_f1"][k] for k in CLASSES]
        ax.bar(np.arange(5) + (i - 1.5) * w, vals, w - 0.02, color=color, label=COND_LABEL[c])
    ax.set_xticks(np.arange(5), [k.replace("_", " ") for k in CLASSES])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (test)", color=INK2)
    ax.set_title(f"Per-class F1 of the final model ({final})", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "per_class_f1.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    places = list(by_place)
    for i, (c, color) in enumerate(zip(CONDS[1:], [ORANGE, AQUA, YELLOW])):
        vals = [by_place[pl]["error_rate_by_snr"][c] for pl in places]
        ax.bar(np.arange(len(places)) + (i - 1) * 0.26, vals, 0.24, color=color, label=COND_LABEL[c])
    ax.set_xticks(np.arange(len(places)), places)
    ax.set_ylabel("Error rate (test mixes)", color=INK2)
    ax.set_ylim(0, max(max(v["error_rate_by_snr"].values()) for v in by_place.values()) * 1.25)
    ax.set_title("Final model errors by background place", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "errors_by_place.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    print(json.dumps(result, indent=1)[:200000])


if __name__ == "__main__":
    main()
