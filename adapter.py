"""SoundSense adapter: give the original UrbanSound8K clips the same properties as the mixed clips.

The mixed clips made by synthesizer.py are 2 s, 16 kHz, mono, 16-bit, peak-normalized WAVs.
The original clips come in 10 sample rates, mono or stereo, 16/24/32-bit, 0.1-4 s long.
This script converts every original clip of the 5 project classes to the mixed-clip format:

  1. resample to 16 kHz, average the channels to mono
  2. cut or pad to exactly 2 s
  3. peak-normalize (same peak level as the mixes), save as 16-bit WAV

Which 2 s piece is kept:
  * If the clip was used in a mix, the same piece at the same position as in the mix (read from the
    synthesizer manifest). The original and mixed version of a clip then differ only by the
    background noise, which keeps the original-vs-mixed comparison fair.
  * Otherwise (clips the mixer did not use), the same rule the mixer uses: a random 2 s piece among
    the loudest ones for long clips, a random position in the 2 s window for short clips, with a
    seed derived from the file name so every run gives the same result.

Folds and splits are unchanged: folds 1-8 train, 9 val, 10 test.

Output: <out>/<split>/<class>/<original name>.wav plus <out>/manifest.csv (one row per clip:
source clip, fold, salience, crop start, position, length in 16 kHz samples, and whether the
piece was taken from a mix).

Usage:
    python adapter.py
    python adapter.py --out D:/soundsense/original
"""
import argparse
import csv
import shutil
import zlib
from pathlib import Path

import numpy as np

import synthesizer as S

ROOT = Path(__file__).resolve().parent
FIELDS = ["file", "split", "label", "source_file", "source_fold", "source_salience",
          "source_crop_start", "event_offset", "event_len", "piece_from_mix"]   # positions in 16 kHz samples


def mix_pieces(manifest):
    """{source file: (crop start, offset, length)} from the first mix that used each source clip."""
    pieces = {}
    if not manifest.exists():
        print(f"No synthesizer manifest at {manifest}: every piece is chosen by the fallback rule")
        return pieces
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_file"] and r["source_file"] not in pieces:
                pieces[r["source_file"]] = (int(r["source_crop_start"]), int(r["event_offset"]), int(r["event_len"]))
    return pieces


def adapt(raw, piece):
    """Cut the 2 s piece, place it in a silent 2 s window, peak-normalize."""
    crop_start, offset, length = piece
    event = raw[crop_start:crop_start + length]
    if len(event) != length:
        raise ValueError(f"piece does not fit the clip ({len(raw)} samples)")
    out = np.zeros(S.WIN, dtype=np.float32)
    out[offset:offset + length] = event
    return S.peak_normalize(out)


def fallback_piece(raw, name, seed):
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    event, crop_start = S.crop_event(raw, rng)
    offset = int(rng.integers(0, S.WIN - len(event) + 1))
    return crop_start, offset, len(event)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--us8k", type=Path, default=ROOT.parent / "UrbanSound8K" / "UrbanSound8K",
                   help="UrbanSound8K root (holds audio/ and metadata/)")
    p.add_argument("--mix-manifest", type=Path, default=ROOT / "data" / "manifest.csv",
                   help="synthesizer manifest, used to cut the same piece as the mixes")
    p.add_argument("--out", type=Path, default=ROOT.parent / "data" / "original",
                   help="output folder; default is outside the repo so the clips are never committed")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--overwrite", action="store_true", help="replace an existing adapted set in --out")
    args = p.parse_args()

    out = args.out
    if (out / "manifest.csv").exists():
        if not args.overwrite:
            raise SystemExit(f"{out} already holds an adapted set; pass --overwrite to replace it")
        for split in S.SPLITS:
            shutil.rmtree(out / split, ignore_errors=True)
        (out / "manifest.csv").unlink()
    out.mkdir(parents=True, exist_ok=True)

    pieces = mix_pieces(args.mix_manifest)
    groups = S.load_events(args.us8k)
    rows, skipped = [], []
    for split in S.SPLITS:
        for label in S.CLASSES:
            for r in sorted(groups[(split, label)], key=lambda r: r["slice_file_name"]):
                name = r["slice_file_name"]
                try:
                    raw = S.load_audio(args.us8k / "audio" / f"fold{r['fold']}" / name)
                    if not np.any(raw):
                        raise ValueError("silent")
                    from_mix = name in pieces
                    piece = pieces[name] if from_mix else fallback_piece(raw, name, args.seed)
                    clip = adapt(raw, piece)
                except ValueError as e:
                    skipped.append(f"{name}: {str(e).split(': ')[-1]}")
                    continue
                path = out / split / label / name
                S.write_wav(path, clip)
                rows.append({"file": path.relative_to(out).as_posix(), "split": split, "label": label,
                             "source_file": name, "source_fold": r["fold"], "source_salience": r["salience"],
                             "source_crop_start": piece[0], "event_offset": piece[1], "event_len": piece[2],
                             "piece_from_mix": int(from_mix)})
            n = sum(1 for x in rows if x["split"] == split and x["label"] == label)
            print(f"{split}/{label}: {n} clips")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done: {len(rows)} clips ({sum(r['piece_from_mix'] for r in rows)} with the same piece as a mix), "
          f"manifest at {out / 'manifest.csv'}")
    if skipped:
        print(f"Skipped {len(skipped)} unreadable clips:")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    main()
