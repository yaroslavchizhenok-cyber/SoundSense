"""SoundSense: build the training, validation and test sets as file lists.

    trainSet1.csv   original clips only            (one per source clip)
    trainSet2.csv   half original + half mixed      (each source clip appears once, in one form)
    trainSet3.csv   mixed clips only                (one per source clip)

All three use exactly the same source clips (folds 1-8), so they differ only in the type of data:
  * a source clip is used only if it has both an adapted original and a mix, giving 800 per class
    (793 for children_playing, whose 7 MS-ADPCM clips cannot be read);
  * where a source was mixed more than once, its first mix is used;
  * in trainSet2 the clips are split per class into two halves, balanced by background place,
    and each half appears in one form only (original or mixed), never both.

Paths in the lists are relative to the data folder that holds original/ and synth/
(by default the "data" folder next to this repo).

Validation and test lists, shared by every model so all results are comparable:

    valSet.csv      fold 9: every adapted original + every mix (random SNR 0-20 dB)
    testSet.csv     fold 10: every adapted original ("clean") + every mix at 20, 10 and 0 dB

The "condition" column says what each row is: clean (original), mixed (train/val mix with a random
SNR), or snr20 / snr10 / snr0 (test mixes), so results can be reported per condition.

Usage:
    python make_sets.py
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CLASSES = ("engine_idling", "air_conditioner", "dog_bark", "children_playing", "street_music")
FIELDS = ["file", "label", "kind", "condition", "source_file", "source_fold", "place", "snr_db"]


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT.parent / "data",
                   help="folder holding original/ (from adapter.py) and synth/ (from synthesizer.py)")
    p.add_argument("--out", type=Path, default=ROOT / "data", help="where the trainSet#.csv lists go")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    orig_manifest = read(args.data / "original" / "manifest.csv")
    mix_manifest = read(args.data / "synth" / "manifest.csv")
    originals = {r["source_file"]: r for r in orig_manifest if r["split"] == "train"}
    mixes = {}
    for r in mix_manifest:
        if r["split"] == "train" and r["source_file"] not in mixes:
            mixes[r["source_file"]] = r

    def as_orig(o):
        return {"file": f"original/{o['file']}", "label": o["label"], "kind": "original", "condition": "clean",
                "source_file": o["source_file"], "source_fold": o["source_fold"], "place": "", "snr_db": ""}

    def as_mix(m):
        cond = f"snr{float(m['snr_db']):g}" if m["split"] == "test" else "mixed"
        return {"file": f"synth/{m['file']}", "label": m["label"], "kind": "mixed", "condition": cond,
                "source_file": m["source_file"], "source_fold": m["source_fold"], "place": m["place"],
                "snr_db": m["snr_db"]}

    def orig_row(src):
        return as_orig(originals[src])

    def mix_row(src):
        return as_mix(mixes[src])

    sets = {1: [], 2: [], 3: []}
    print(f"{'class':18s} {'set1':>5s} {'set2 orig+mix':>14s} {'set3':>5s}")
    for ci, label in enumerate(CLASSES):
        sources = sorted(s for s, o in originals.items() if o["label"] == label and s in mixes)
        sets[1] += [orig_row(s) for s in sources]
        sets[3] += [mix_row(s) for s in sources]

        # set 2: within each place, shuffle and alternate original/mixed -> exact halves, balanced by place
        rng = np.random.default_rng([args.seed, ci])
        by_place = defaultdict(list)
        for s in sources:
            by_place[mixes[s]["place"]].append(s)
        ordered = []
        for place in sorted(by_place):
            group = by_place[place]
            ordered += [group[i] for i in rng.permutation(len(group))]
        half = [orig_row(s) if i % 2 == 0 else mix_row(s) for i, s in enumerate(ordered)]
        sets[2] += half
        n_orig = sum(r["kind"] == "original" for r in half)
        print(f"{label:18s} {len(sources):5d} {n_orig:6d} + {len(half) - n_orig:<6d} {len(sources):5d}")

    lists = {f"trainSet{k}": rows for k, rows in sets.items()}
    for split, name in (("val", "valSet"), ("test", "testSet")):
        lists[name] = ([as_orig(o) for o in orig_manifest if o["split"] == split]
                       + [as_mix(m) for m in mix_manifest if m["split"] == split])
        counts = {}
        for r in lists[name]:
            counts[r["condition"]] = counts.get(r["condition"], 0) + 1
        print(f"{name}: " + ", ".join(f"{c} {n}" for c, n in counts.items()))

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in lists.items():
        missing = [r["file"] for r in rows if not (args.data / r["file"]).exists()]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} listed files do not exist, e.g. {missing[0]}")
        path = args.out / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{path.name}: {len(rows)} clips")


if __name__ == "__main__":
    main()
