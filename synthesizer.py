"""SoundSense synthesizer: mix UrbanSound8K events with the team's own background recordings.

Inputs
  * UrbanSound8K, 5 classes (CLASSES below). Folds are kept as they are: 1-8 train, 9 val, 10 test,
    so every mix inherits the fold of its source clip.
  * The team's background recordings in noise/ (next to this script), one file per place (house, kitchen, ...).
    WAV is read directly; another format (e.g. .m4a) without a .wav twin is decoded with ffmpeg.

How the recordings are shared out (every recording in every split, evenly, no leakage):
  * 2 s are cut from both ends of each recording (record-button handling noise).
  * The rest is cut into 4 s blocks, dealt out in a fixed repeating pattern over the whole
    recording: ~70% of the blocks to train, ~15% to val, ~15% to test. A background crop never
    crosses a block boundary, so no stretch of background audio is in two splits.
  * Within every split x class, each place is used the same number of times (+-1), and the
    random SNRs are stratified, so they cover 0-20 dB evenly for every class.

Output: 2 s, 16 kHz, mono, peak-normalized 16-bit WAVs plus <out>/manifest.csv (one row per clip:
source clip, fold, place, background position, SNR, seed; positions are 16 kHz sample indices, so
any mix can be rebuilt exactly).

    split   folds   per class
    train   1-8     800 mixes, SNR 0-20 dB
    val     9       100 mixes, SNR 0-20 dB
    test    10      100 events x {20, 10, 0} dB = 300

In the test split each event is mixed with the same crop, position and background at every SNR,
so the only thing that changes between the 20/10/0 dB versions is the noise level.

SNR = 10*log10(P_event / P_background): P_background is the mean power of the 2 s background crop,
P_event the mean power of the event's active 50 ms frames (within 20 dB of its loudest frame), so a
single short bark is not under-rated. Salience-1 (foreground) clips are used before salience-2.

Requires only NumPy (plus ffmpeg on PATH for non-WAV recordings).

Usage:
    python synthesizer.py
    python synthesizer.py --train-per-class 20 --val-per-class 5 --test-per-class 5
"""
import argparse
import csv
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SR = 16000
WIN = 2 * SR                      # 2 s window
PEAK = 0.9                        # peak level after normalization
BG_TRIM = 2 * SR                  # samples cut from both ends of each background recording
BG_BLOCK = 4 * SR                 # background block length for the train/val/test time split
BG_PATTERN = ("train", "train", "train", "test", "train", "val", "train")   # 5/7 train, 1/7 val, 1/7 test
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac"}
CLASSES = ("engine_idling", "air_conditioner", "dog_bark", "children_playing", "street_music")
SPLITS = ("train", "val", "test")
SPLIT_FOLDS = {"train": set(range(1, 9)), "val": {9}, "test": {10}}


# ---------------------------------------------------------------- audio I/O (NumPy only)

def read_wav(path):
    """Read PCM 8/16/24/32-bit or float 32/64-bit WAV (incl. WAVE_FORMAT_EXTENSIBLE) as mono float32."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            fmt = data[pos + 8:pos + 8 + size]
        elif cid == b"data":
            pcm = data[pos + 8:pos + 8 + size]      # truncated files: slice stops at end of file
            break
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError(f"{path}: missing fmt or data chunk")

    tag, channels, sr = struct.unpack("<HHI", fmt[:8])
    block_align = struct.unpack("<H", fmt[12:14])[0]
    if tag == 0xFFFE and len(fmt) >= 26:          # extensible: real format is in the sub-format GUID
        tag = struct.unpack("<H", fmt[24:26])[0]
    width = block_align // channels
    pcm = pcm[:len(pcm) // block_align * block_align]

    if tag == 3 and width in (4, 8):
        x = np.frombuffer(pcm, "<f4" if width == 4 else "<f8").astype(np.float32)
    elif tag == 1 and width == 1:
        x = (np.frombuffer(pcm, np.uint8).astype(np.float32) - 128) / 128
    elif tag == 1 and width == 2:
        x = np.frombuffer(pcm, "<i2").astype(np.float32) / 2 ** 15
    elif tag == 1 and width == 3:
        b = np.frombuffer(pcm, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float32) / 2 ** 23
    elif tag == 1 and width == 4:
        x = (np.frombuffer(pcm, "<i4") / 2 ** 31).astype(np.float32)
    else:
        raise ValueError(f"{path}: unsupported WAV format (tag={tag}, {8 * width}-bit)")
    return x.reshape(-1, channels).mean(axis=1), sr


def write_wav(path, x, sr=SR):
    pcm = (np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes()
    header = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
              + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
              + b"data" + struct.pack("<I", len(pcm)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + pcm)


def resample(x, sr_in, sr_out=SR):
    """Band-limited FFT resampling (keeps only frequencies below the new Nyquist)."""
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    spec = np.fft.rfft(x)
    out = np.zeros(n_out // 2 + 1, dtype=spec.dtype)
    k = min(len(spec), len(out))
    out[:k] = spec[:k]
    return (np.fft.irfft(out, n_out) * (n_out / len(x))).astype(np.float32)


def load_audio(path):
    """Any WAV -> 16 kHz mono float32. Other formats go through ffmpeg."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return resample(*read_wav(path))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit(f"{path.name}: install ffmpeg, or save a WAV copy next to it ({path.stem}.wav)")
    pcm = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, "<i2").astype(np.float32) / 2 ** 15


# ---------------------------------------------------------------- the mixer

def frame_power(x, frame=SR // 20):
    n = len(x) // frame
    if n == 0:
        return np.array([np.mean(x ** 2)])
    return np.mean(x[:n * frame].reshape(n, frame) ** 2, axis=1)


def active_power(x, floor_db=20):
    p = frame_power(x)
    return p[p >= p.max() * 10 ** (-floor_db / 10)].mean()


def crop_event(x, rng):
    """Events longer than 2 s: random 2 s crop among those with >= half the loudest crop's energy."""
    if len(x) <= WIN:
        return x, 0
    starts = np.arange(0, len(x) - WIN + 1, SR // 10)
    energy = np.array([np.mean(x[s:s + WIN] ** 2) for s in starts])
    s = int(rng.choice(starts[energy >= 0.5 * energy.max()]))
    return x[s:s + WIN], s


def mix(event, offset, background, snr_db):
    """Place `event` at `offset` in the 2 s background at the given SNR, then peak-normalize."""
    gain = np.sqrt(np.mean(background ** 2) * 10 ** (snr_db / 10) / active_power(event))
    out = background.copy()
    out[offset:offset + len(event)] += gain * event
    return peak_normalize(out)


def peak_normalize(x):
    peak = np.max(np.abs(x))
    return x * (PEAK / peak) if peak > 0 else x


# ---------------------------------------------------------------- backgrounds

def split_segments(n_samples):
    """{split: [(start, end)]} sample ranges of one recording, from the 4 s block pattern.
    Consecutive blocks of the same split are merged so crops can start anywhere inside them."""
    usable = n_samples - 2 * BG_TRIM
    n_blocks = usable // BG_BLOCK
    if n_blocks < len(BG_PATTERN):
        raise SystemExit(f"Recording too short: needs >= {(len(BG_PATTERN) * BG_BLOCK + 2 * BG_TRIM) / SR:.0f} s")
    segs = {s: [] for s in SPLITS}
    for b in range(n_blocks):
        split = BG_PATTERN[b % len(BG_PATTERN)]
        start = BG_TRIM + b * BG_BLOCK
        end = BG_TRIM + usable if b == n_blocks - 1 else start + BG_BLOCK   # last block takes the remainder
        if segs[split] and segs[split][-1][1] == start:
            segs[split][-1] = (segs[split][-1][0], end)
        else:
            segs[split].append((start, end))
    return segs


def load_backgrounds(bg_dir):
    """{place: (path, audio, {split: segments})}, one place per recording (file stem)."""
    files = {}
    for p in sorted(bg_dir.iterdir()):
        if p.suffix.lower() in AUDIO_EXT and (p.stem not in files or p.suffix.lower() == ".wav"):
            files[p.stem] = p                       # prefer the WAV when both exist
    if not files:
        raise SystemExit(f"No recordings found in {bg_dir}")
    places = {}
    print(f"Backgrounds ({bg_dir}): seconds per split")
    for place, path in files.items():
        audio = load_audio(path)
        segs = split_segments(len(audio))
        places[place] = (path, audio, segs)
        secs = {s: sum(e - a for a, e in segs[s]) / SR for s in SPLITS}
        print(f"  {place:12s} " + "  ".join(f"{s} {secs[s]:5.1f}" for s in SPLITS))
    return places


def crop_background(entry, split, rng, min_db=-90, tries=50):
    """Random 2 s crop from this place's `split` segments. Rejects only digital silence
    (a quiet room on a phone mic sits around -65 to -70 dBFS, which is real background)."""
    path, audio, segs = entry
    ranges = segs[split]
    n_starts = np.array([e - a - WIN + 1 for a, e in ranges], dtype=float)
    for _ in range(tries):
        k = rng.choice(len(ranges), p=n_starts / n_starts.sum())
        start = ranges[k][0] + int(rng.integers(0, n_starts[k]))
        crop = audio[start:start + WIN]
        if 10 * np.log10(np.mean(crop ** 2) + 1e-12) > min_db:
            return crop, start
    raise RuntimeError(f"{path.name}: no non-silent background crop in {split}")


def balanced(items, n, rng):
    """n items with every item used n/len(items) times (+-1), in random order."""
    return [items[i % len(items)] for i in rng.permutation(n)]


def stratified(lo, hi, n, rng):
    """n values covering [lo, hi] evenly: one random value per equal-width bin, bins shuffled."""
    return lo + (hi - lo) * (rng.permutation(n) + rng.random(n)) / n


# ---------------------------------------------------------------- event sources

def load_events(us8k):
    """{(split, class): [metadata rows]}"""
    groups = {(s, c): [] for s in SPLITS for c in CLASSES}
    with open(us8k / "metadata" / "UrbanSound8K.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["class"] not in CLASSES:
                continue
            split = next(s for s, folds in SPLIT_FOLDS.items() if int(r["fold"]) in folds)
            groups[(split, r["class"])].append(r)
    return groups


def order_sources(rows, rng, salience1_only):
    s1 = [r for r in rows if r["salience"] == "1"]
    s2 = [r for r in rows if r["salience"] == "2"]
    rng.shuffle(s1)
    rng.shuffle(s2)
    return s1 if salience1_only else s1 + s2


def next_source(sources, j, audio_dir, limit):
    """Load sources[j], skipping unreadable or silent clips; returns (row, audio, next j)."""
    while True:
        if j >= limit:
            raise SystemExit("Too many unusable source clips")
        src = sources[j % len(sources)]
        j += 1
        try:
            raw = load_audio(audio_dir / f"fold{src['fold']}" / src["slice_file_name"])
        except ValueError as e:     # e.g. 7 MS-ADPCM children_playing clips (fsID 36429)
            print(f"  skip {src['slice_file_name']}: {str(e).split(': ')[-1]}")
            continue
        if np.any(raw):
            return src, raw, j
        print(f"  skip {src['slice_file_name']}: silent")


# ---------------------------------------------------------------- generation

FIELDS = ["file", "split", "label", "snr_db", "source_file", "source_fold", "source_salience",
          "source_crop_start", "event_offset", "event_len",
          "place", "bg_file", "bg_start", "seed"]             # positions/lengths in samples at 16 kHz


def clip_rng(seed, split, label, i):
    clip_seed = int(np.random.SeedSequence([seed, SPLITS.index(split), CLASSES.index(label), i])
                    .generate_state(1)[0])
    return np.random.default_rng(clip_seed), clip_seed


def generate(args):
    out = args.out
    if (out / "manifest.csv").exists():
        if not args.overwrite:
            raise SystemExit(f"{out} already holds a synthesized set; pass --overwrite to replace it")
        for split in SPLITS:
            shutil.rmtree(out / split, ignore_errors=True)
        (out / "manifest.csv").unlink()
    out.mkdir(parents=True, exist_ok=True)

    backgrounds = load_backgrounds(args.backgrounds)
    places = sorted(backgrounds)
    events = load_events(args.us8k)
    audio_dir = args.us8k / "audio"
    per_class = {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class}
    rows = []

    def save(x, split, label, name, place, bg_start, **info):
        path = out / split / label / name
        write_wav(path, x)
        rows.append({"file": path.relative_to(out).as_posix(), "split": split, "label": label,
                     "place": place, "bg_file": backgrounds[place][0].name, "bg_start": bg_start, **info})
        if len(rows) % 500 == 0:
            print(f"  {len(rows)} clips written")

    for split in args.splits:
        for ci, label in enumerate(CLASSES):
            plan_rng = np.random.default_rng([args.seed, SPLITS.index(split), ci])
            n = per_class[split]
            place_seq = balanced(places, n, plan_rng)
            sources = order_sources(events[(split, label)], plan_rng, args.salience1_only)
            if not sources:
                raise SystemExit(f"No source clips for {label} in {split}")
            fixed_snrs = args.test_snrs if split == "test" else None
            snr_seq = None if fixed_snrs else np.round(stratified(*args.snr_range, n, plan_rng), 2)
            print(f"{split}/{label}: {n} events from {len(sources)} clips"
                  + (f" ({n - len(sources)} reused with a new crop/background)" if n > len(sources) else ""))

            j = 0
            for i in range(n):
                src, raw, j = next_source(sources, j, audio_dir, limit=n + 2 * len(sources))
                rng, clip_seed = clip_rng(args.seed, split, label, i)
                event, crop_start = crop_event(raw, rng)
                offset = int(rng.integers(0, WIN - len(event) + 1))
                place = place_seq[i]
                bg, bg_start = crop_background(backgrounds[place], split, rng)
                stem = Path(src["slice_file_name"]).stem
                for snr in fixed_snrs or [float(snr_seq[i])]:
                    name = f"{label}_{i:04d}_{stem}_{place}" + (f"_snr{snr:g}" if fixed_snrs else "") + ".wav"
                    save(mix(event, offset, bg, snr), split, label, name, place, bg_start,
                         snr_db=snr, source_file=src["slice_file_name"], source_fold=src["fold"],
                         source_salience=src["salience"], source_crop_start=crop_start,
                         event_offset=offset, event_len=len(event), seed=clip_seed)

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")   # LF, matches .gitattributes
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done: {len(rows)} clips, manifest at {out / 'manifest.csv'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--us8k", type=Path, default=ROOT.parent / "UrbanSound8K" / "UrbanSound8K",
                   help="UrbanSound8K root (holds audio/ and metadata/); default: next to the repo folder")
    p.add_argument("--backgrounds", type=Path, default=ROOT / "noise",
                   help="folder with the team's recordings, one file per place")
    p.add_argument("--out", type=Path, default=ROOT.parent / "data" / "synth",
                   help="output folder; default is outside the repo so the clips are never committed")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    p.add_argument("--train-per-class", type=int, default=800)
    p.add_argument("--val-per-class", type=int, default=100)
    p.add_argument("--test-per-class", type=int, default=100, help="events; each is mixed at every --test-snrs")
    p.add_argument("--snr-range", type=float, nargs=2, default=[0.0, 20.0], metavar=("LO", "HI"),
                   help="SNR range (dB) for train and val, covered evenly")
    p.add_argument("--test-snrs", type=float, nargs="+", default=[20.0, 10.0, 0.0])
    p.add_argument("--salience1-only", action="store_true",
                   help="use only foreground (salience 1) clips, reusing them to reach the counts")
    p.add_argument("--overwrite", action="store_true", help="replace an existing synthesized set in --out")
    generate(p.parse_args())


if __name__ == "__main__":
    main()
