"""SoundSense features: the one spectrogram function used by training, evaluation and the app.

A 2 s, 16 kHz mono clip becomes a 64 x 201 log-mel spectrogram ("the grid of numbers"):
64 mel bands from 50 to 8,000 Hz (rows) x 201 time frames, one every 10 ms (columns).

Settings (from the project plan):
    window 25 ms (400 samples, Hann), hop 10 ms (160 samples), FFT size 400,
    frames centred with reflect padding, power spectrum, 64 HTK-mel bands, 50-8,000 Hz,
    then 10*log10(power) (dB, power floored at 1e-10), and finally every value is raised to at least
    (clip maximum - 80 dB). The 80 dB floor stops digital silence (zero padding of short original
    clips) from showing up as -100 dB, which the mixed clips never have.

This NumPy version follows torchaudio's defaults, so in PyTorch the same features are
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160,
                                         f_min=50, f_max=8000, n_mels=64, power=2.0, center=True,
                                         pad_mode="reflect", norm=None, mel_scale="htk")
    followed by torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80), applied to ONE clip at
    a time, shaped (1, 64, 201) or (64, 201). On a batch shaped (B, 64, 201) torchaudio would take the
    maximum over the whole batch; use (B, 1, 64, 201) so the floor is per clip.
Check once that both give the same numbers (to about 1e-3 dB) before mixing the two.

Scaling: every spectrogram is scaled as (x - mean) / std, with mean and std computed over all
values of all spectrograms of one training set (training clips only, never val/test).
Running this file computes them for trainSet1-3 and saves data/stats_trainSet#.json.

Usage:
    python features.py                      # compute and save the scaling numbers
    python features.py --data D:/soundsense/data
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from synthesizer import read_wav, resample

ROOT = Path(__file__).resolve().parent
SR = 16000
CLIP = 2 * SR             # 2 s
N_FFT = 400               # 25 ms
HOP = 160                 # 10 ms
N_MELS = 64
F_MIN, F_MAX = 50.0, 8000.0
AMIN = 1e-10              # power floor before the log
TOP_DB = 80.0             # values are kept within 80 dB of the clip's maximum
SETTINGS = {"sample_rate": SR, "clip_samples": CLIP, "n_fft": N_FFT, "win_length": N_FFT, "hop_length": HOP,
            "n_mels": N_MELS, "f_min": F_MIN, "f_max": F_MAX, "window": "hann (periodic)", "center": True,
            "pad_mode": "reflect", "power": 2.0, "mel_scale": "htk", "norm": None, "log": "10*log10(max(power, 1e-10))", "top_db": TOP_DB}


def _mel_filterbank():
    """HTK triangular mel filters, computed the same way as torchaudio.functional.melscale_fbanks."""
    hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel_to_hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    all_freqs = np.linspace(0, SR // 2, N_FFT // 2 + 1)
    f_pts = mel_to_hz(np.linspace(hz_to_mel(F_MIN), hz_to_mel(F_MAX), N_MELS + 2))
    f_diff = np.diff(f_pts)
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up))          # (n_freqs, n_mels)


_FB = _mel_filterbank()
_WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)   # periodic Hann, like torch


def log_mel(clip):
    """2 s, 16 kHz mono float clip -> (64, 201) float32 log-mel spectrogram in dB."""
    clip = np.asarray(clip, dtype=np.float64)
    if clip.shape != (CLIP,):
        raise ValueError(f"expected {CLIP} samples (2 s at 16 kHz), got {clip.shape}")
    x = np.pad(clip, N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    power = np.abs(np.fft.rfft(x[idx] * _WINDOW, axis=1)) ** 2          # (201, 201)
    mel = power @ _FB                                                     # (201 frames, 64)
    db = 10.0 * np.log10(np.maximum(mel, AMIN))
    return np.maximum(db, db.max() - TOP_DB).T.astype(np.float32)


def load_clip(path):
    """Read a prepared clip (output of synthesizer.py / adapter.py) as 16 kHz mono float."""
    x, sr = read_wav(path)
    return resample(x, sr, SR) if sr != SR else x


def load_stats(path):
    with open(path, encoding="utf-8") as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]


def scale(features, mean, std):
    return (features - mean) / std


def compute_stats(list_csv, data_dir):
    """Mean and std over every value of every spectrogram in one training list."""
    total = total_sq = 0.0
    count = 0
    with open(list_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows, 1):
        feat = log_mel(load_clip(data_dir / r["file"])).astype(np.float64)
        total += feat.sum()
        total_sq += (feat ** 2).sum()
        count += feat.size
        if i % 1000 == 0:
            print(f"  {i}/{len(rows)}")
    mean = total / count
    std = float(np.sqrt(total_sq / count - mean ** 2))
    return {"mean": float(mean), "std": std, "clips": len(rows), "values": count}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT.parent / "data",
                   help="folder holding original/ and synth/ (paths in the lists are relative to it)")
    p.add_argument("--lists", type=Path, default=ROOT / "data", help="folder with trainSet#.csv")
    args = p.parse_args()

    for k in (1, 2, 3):
        name = f"trainSet{k}"
        print(f"{name}:")
        stats = compute_stats(args.lists / f"{name}.csv", args.data)
        stats.update(train_list=f"{name}.csv", features=SETTINGS)
        out = args.lists / f"stats_{name}.json"
        out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8", newline="\n")   # LF
        print(f"  mean {stats['mean']:.3f} dB, std {stats['std']:.3f} dB -> {out.name}")


if __name__ == "__main__":
    main()
