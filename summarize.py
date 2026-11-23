"""Collect every run into one results table for the experiment log and the report (Part C).

    python summarize.py

Writes runs/results.csv and runs/results.md (a Markdown table you can paste into the report).
Test columns stay empty until evaluate.py has been run for that model.
"""
import csv
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
CONDITIONS = ["clean", "snr20", "snr10", "snr0", "overall"]


def main():
    rows = []
    for summary_file in sorted(RUNS.glob("*/summary.json")):
        s = json.loads(summary_file.read_text())
        if s.get("limit"):
            continue                       # skip debug runs (--limit)
        row = {
            "run": s["run"],
            "model": s["model"] + (" + aug" if s["augment"] else ""),
            "train_set": s["train_set"],
            "best_epoch": s["best_epoch"],
            "val_macro_f1": round(s["best_val_macro_f1"], 4),
        }
        test_file = summary_file.parent / "test_results.json"
        test = json.loads(test_file.read_text()) if test_file.exists() else {}
        for c in CONDITIONS:
            row[f"test_{c}"] = round(test[c]["macro_f1"], 4) if c in test else ""
        rows.append(row)

    if not rows:
        print("No finished runs in runs/ yet.")
        return

    with open(RUNS / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(r[h]) for h in header) + " |" for r in rows]
    (RUNS / "results.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved {RUNS / 'results.csv'} and {RUNS / 'results.md'}")


if __name__ == "__main__":
    main()
