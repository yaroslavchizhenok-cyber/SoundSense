"""Data loading for SoundSense models (Part C).

Wraps the team's features.py and follows data_manual.md:
  - reads ONLY the lists in data/ (trainSet1-3.csv, valSet.csv, testSet.csv)
  - computes every log-mel spectrogram once with features.log_mel and caches it on disk
    (UNSCALED, because the scaling numbers depend on which training set a model uses)
  - scales with stats_trainSet#.json of the training set the model is trained on
  - augmentation (time shift + SpecAugment) for training batches only
"""
import csv
from pathlib import Path

import numpy as np
import torch

import features as F  # the team's feature function (must stay the only one)

REPO = Path(__file__).resolve().parent
LISTS_DIR = REPO / "data"                      # lists + stats, inside the repo
DEFAULT_DATA_DIR = REPO.parent / "data"        # generated audio, next to the repo
DEFAULT_CACHE_DIR = REPO.parent / "cache"      # cached spectrograms, next to the repo (never in git)

# Fixed class order from data_manual.md: the position is the class id.
CLASSES = ["engine_idling", "air_conditioner", "dog_bark", "children_playing", "street_music"]
N_MELS, N_FRAMES = 64, 201
TEST_CONDITIONS = ["clean", "snr20", "snr10", "snr0"]


def read_list(name):
    """Rows of data/<name>.csv as dictionaries."""
    with open(LISTS_DIR / f"{name}.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_list(name, data_dir=DEFAULT_DATA_DIR, cache_dir=DEFAULT_CACHE_DIR):
    """Unscaled log-mels for every clip in data/<name>.csv.

    Returns X (N, 1, 64, 201) float32, y (N,) int64, rows (list of dicts from the CSV).
    The first call computes and caches (~13 ms per clip); later calls load in seconds.
    """
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    rows = read_list(name)
    cache_file = cache_dir / f"{name}.npz"

    if cache_file.exists():
        cached = np.load(cache_file)
        if len(cached["y"]) == len(rows):
            return cached["X"], cached["y"], rows
        print(f"[data] cache {cache_file.name} does not match the list, rebuilding")

    print(f"[data] computing log-mels for {name} ({len(rows)} clips), this happens once...")
    X = np.zeros((len(rows), 1, N_MELS, N_FRAMES), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        X[i, 0] = F.log_mel(F.load_clip(data_dir / r["file"]))
        y[i] = CLASSES.index(r["label"])
        if (i + 1) % 500 == 0:
            print(f"[data]   {i + 1}/{len(rows)}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, X=X, y=y)
    return X, y, rows


def stats_file_name(train_set):
    return f"stats_trainSet{train_set}.json"


def load_stats(train_set):
    """(mean, std, file name) of the training set the model is trained on."""
    name = stats_file_name(train_set)
    mean, std = F.load_stats(LISTS_DIR / name)
    return float(mean), float(std), name


def scale(X, mean, std):
    """(x - mean) / std as a float32 torch tensor (same formula as features.scale)."""
    X = X.astype(np.float32, copy=True)
    X -= np.float32(mean)
    X /= np.float32(std)
    return torch.from_numpy(X)


# ---------------------------------------------------------------- augmentation
# Training only, on the fly, identical for every training set (data_manual rule 6).
# No gain augmentation: every clip is peak-normalized, so volume never varies.

def time_shift(x, max_shift=40):
    """Shift each spectrogram left/right by up to max_shift frames (40 frames = 0.4 s).
    The gap that opens up is filled with the clip's quietest value (silence)."""
    out = x.clone()
    shifts = torch.randint(-max_shift, max_shift + 1, (x.size(0),))
    for i, s in enumerate(shifts.tolist()):
        if s == 0:
            continue
        floor = x[i].min()
        out[i] = torch.roll(x[i], shifts=s, dims=-1)
        if s > 0:
            out[i, ..., :s] = floor
        else:
            out[i, ..., s:] = floor
    return out


def spec_augment(x, freq_width=8, time_width=20, n_freq_masks=1, n_time_masks=2):
    """SpecAugment: blank out random horizontal (frequency) and vertical (time) stripes,
    so the model cannot rely on any single band or moment. Masked values are set to 0,
    which after scaling is the training set's average level."""
    out = x.clone()
    B, _, n_freq, n_time = x.shape
    for i in range(B):
        for _ in range(n_freq_masks):
            w = int(torch.randint(0, freq_width + 1, (1,)))
            f0 = int(torch.randint(0, n_freq - w + 1, (1,)))
            out[i, :, f0:f0 + w, :] = 0.0
        for _ in range(n_time_masks):
            w = int(torch.randint(0, time_width + 1, (1,)))
            t0 = int(torch.randint(0, n_time - w + 1, (1,)))
            out[i, :, :, t0:t0 + w] = 0.0
    return out


def augment(x):
    return spec_augment(time_shift(x))
