"""SoundSense adapter for the app: turn an uploaded file or live microphone audio into model input.

It applies exactly the training preprocessing, so the model sees the same kind of input it learned on:

    any audio -> 16 kHz mono -> 2 s windows every 1 s -> silence gate -> peak-normalize each window
              -> 64 x 201 log-mel spectrogram (features.log_mel) -> scale with the training set's numbers

Input checks (raise AudioError with a message that can be shown to the user):
  * file type: .wav is read directly; .mp3/.m4a/.ogg/.flac/.webm need ffmpeg on the server
    (browsers usually record microphone audio as .webm, so either install ffmpeg or send raw samples);
  * unreadable or empty audio;
  * files longer than MAX_SECONDS.

Silence gate: a window whose loudness (RMS) is below SILENCE_DB dBFS *before* normalization is marked
silent and not classified; otherwise normalization would blow quiet room noise up to full volume.
SILENCE_DB depends on the microphone; tune it on the demo laptop.

Usage from the app:
    from adapter_for_app import prepare_file, prepare_samples, LiveStream, AudioError

    result = prepare_file("upload.wav", "data/stats_trainSet2.json")
    result.features   # (n_windows, 64, 201) float32, ready for the model (add a channel axis if needed)
    result.starts     # start time of each window in seconds
    result.silent     # True where the window was too quiet; skip those predictions

    live = LiveStream("data/stats_trainSet2.json")
    for window in live.push(samples, sample_rate):   # call with every chunk from the microphone
        ...                                          # one PreparedAudio per new 2 s window (every 1 s)

Run directly to test it on a file:
    python adapter_for_app.py some_recording.wav --stats data/stats_trainSet1.json
"""
import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from features import CLIP, SR, load_stats, log_mel, scale
from synthesizer import PEAK, read_wav, resample

WINDOW_HOP = SR            # a new 2 s window every 1 s
SILENCE_DB = -50.0         # window RMS (dBFS, before normalization) below this counts as silence
MAX_SECONDS = 600          # longest upload accepted
FFMPEG_TYPES = {".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}


class AudioError(Exception):
    """Problem with the user's audio; the message is safe to show in the app."""


@dataclass
class PreparedAudio:
    features: np.ndarray   # (n, 64, 201) float32, scaled
    starts: np.ndarray     # (n,) window start times in seconds
    silent: np.ndarray     # (n,) bool, True = too quiet, do not classify
    duration: float        # seconds of audio received


# ---------------------------------------------------------------- loading

def load_file(path):
    """Any supported audio file -> 16 kHz mono float32."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".wav":
        try:
            x, sr = read_wav(path)
        except (ValueError, OSError) as e:     # no details: they contain server paths
            raise AudioError("Could not read this WAV file; it may be damaged or use an unusual format.") from e
        return to_16k_mono(x, sr)
    if ext in FFMPEG_TYPES:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise AudioError(f"{ext} files are not supported on this server; please upload a .wav file.")
        run = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-t", str(MAX_SECONDS + 1),
                              "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True)
        if run.returncode != 0:
            raise AudioError("Could not decode this audio file.")
        return np.frombuffer(run.stdout, "<i2").astype(np.float32) / 2 ** 15
    raise AudioError(f"Unsupported file type '{ext or 'none'}'. Please upload a .wav file.")


def to_16k_mono(samples, sample_rate):
    """Raw samples (1-D mono, or 2-D shaped (n, channels)) at any rate -> 16 kHz mono float32."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1:
        raise AudioError("Audio must be mono or (samples, channels).")
    if np.issubdtype(np.asarray(samples).dtype, np.integer):
        x = x / np.iinfo(np.asarray(samples).dtype).max
    return resample(x, int(sample_rate), SR)


# ---------------------------------------------------------------- windows and features

def split_windows(audio):
    """2 s windows every 1 s. Shorter audio is zero-padded to one window; a final window is aligned
    to the end so the last second is never dropped."""
    if len(audio) <= CLIP:
        return [0], [np.pad(audio, (0, CLIP - len(audio)))]
    starts = list(range(0, len(audio) - CLIP + 1, WINDOW_HOP))
    if starts[-1] != len(audio) - CLIP:
        starts.append(len(audio) - CLIP)
    return starts, [audio[s:s + CLIP] for s in starts]


def window_features(window, mean, std):
    """One 2 s window -> (scaled 64 x 201 features, is_silent)."""
    rms_db = 10 * np.log10(np.mean(window ** 2) + 1e-12)
    silent = rms_db < SILENCE_DB
    peak = np.max(np.abs(window))
    normalized = window * (PEAK / peak) if peak > 0 else window
    return scale(log_mel(normalized), mean, std).astype(np.float32), bool(silent)


def prepare_audio(audio, stats):
    """16 kHz mono float audio -> PreparedAudio. `stats` is a stats_trainSet#.json path or (mean, std)."""
    mean, std = load_stats(stats) if isinstance(stats, (str, Path)) else stats
    if len(audio) == 0:
        raise AudioError("The recording is empty.")
    if len(audio) > MAX_SECONDS * SR:
        raise AudioError(f"The recording is longer than {MAX_SECONDS // 60} minutes; please upload a shorter one.")
    starts, windows = split_windows(audio)
    feats, silent = zip(*(window_features(w, mean, std) for w in windows))
    return PreparedAudio(np.stack(feats), np.array(starts) / SR, np.array(silent), len(audio) / SR)


def prepare_file(path, stats):
    return prepare_audio(load_file(path), stats)


def prepare_samples(samples, sample_rate, stats):
    return prepare_audio(to_16k_mono(samples, sample_rate), stats)


# ---------------------------------------------------------------- live microphone

class LiveStream:
    """Collects microphone chunks and returns a prepared 2 s window every 1 s of new audio."""

    def __init__(self, stats):
        self.mean, self.std = load_stats(stats) if isinstance(stats, (str, Path)) else stats
        self.buffer = np.zeros(0, dtype=np.float32)
        self.received = 0          # 16 kHz samples received so far
        self.next_end = CLIP       # sample position where the next window ends

    def push(self, samples, sample_rate):
        """Add a chunk; returns a list of PreparedAudio (one window each), usually empty or one item."""
        chunk = to_16k_mono(samples, sample_rate)
        self.buffer = np.concatenate([self.buffer, chunk])
        self.received += len(chunk)
        out = []
        while self.received >= self.next_end:
            end_in_buffer = len(self.buffer) - (self.received - self.next_end)
            window = self.buffer[end_in_buffer - CLIP:end_in_buffer]
            feat, silent = window_features(window, self.mean, self.std)
            out.append(PreparedAudio(feat[None], np.array([(self.next_end - CLIP) / SR]),
                                     np.array([silent]), CLIP / SR))
            self.next_end += WINDOW_HOP
        self.buffer = self.buffer[-CLIP:]          # keep only what the next window can need
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path)
    p.add_argument("--stats", type=Path, default=Path(__file__).resolve().parent / "data" / "stats_trainSet1.json")
    args = p.parse_args()
    try:
        r = prepare_file(args.file, args.stats)
    except AudioError as e:
        raise SystemExit(f"AudioError: {e}")
    print(f"{args.file.name}: {r.duration:.1f} s -> {len(r.starts)} windows, features {r.features.shape}, "
          f"{int(r.silent.sum())} silent")
    for t, s in zip(r.starts, r.silent):
        print(f"  {t:6.1f} s  {'silent' if s else 'classify'}")


if __name__ == "__main__":
    main()
