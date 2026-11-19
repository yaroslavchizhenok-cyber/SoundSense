# Data manual for model building

This manual covers everything you need to train and evaluate SoundSense models: how to get the data, what each file contains, and the rules that keep the three experiments comparable. The data is ready; your part starts at loading the lists in `data/`.

## 1. What the data is

Five UrbanSound8K classes, always in this order (use the position as the class id):

| Id | Class |
|---|---|
| 0 | `engine_idling` |
| 1 | `air_conditioner` |
| 2 | `dog_bark` |
| 3 | `children_playing` |
| 4 | `street_music` |

Every clip, original or mixed, has the same format: **2 s, 16 kHz, mono, 16-bit WAV, peak-normalized to 0.9**.

- **Original clips** are UrbanSound8K clips converted to that format by `adapter.py`.
- **Mixed clips** are the same events mixed by `synthesizer.py` over the team's own background recordings (house, kitchen, lecture, room, university) at a chosen signal-to-noise ratio (SNR).

An original clip and its mix contain the same 2 s piece of the event at the same position; they differ only by the background.

Folds are fixed by UrbanSound8K and never reshuffled: **folds 1–8 train, fold 9 validation, fold 10 test**.

## 2. Getting the audio

The audio is not in the repo (about 700 MB). Build it once, from inside the `SoundSense` folder, with UrbanSound8K extracted at `../UrbanSound8K/UrbanSound8K` (the folder holding `audio/` and `metadata/`):

```
python synthesizer.py     # mixed clips   -> ../data/synth/     (~2 min, 6,000 files)
python adapter.py         # original clips -> ../data/original/ (~1 min, 4,993 files)
```

Both are deterministic (fixed seed), so everyone gets identical files. If UrbanSound8K is elsewhere, add `--us8k <path>`. Only NumPy is needed. Write the output outside OneDrive if you can (`--out`), since OneDrive will try to sync thousands of files.

The lists and scaling numbers in `data/` are already built and committed. Rebuild them only if the audio changes:

```
python make_sets.py       # data/trainSet1-3.csv, valSet.csv, testSet.csv (seconds)
python features.py        # data/stats_trainSet1-3.json (~1 min)
```

Resulting layout:

```
INHA_projects/AI(SoundSense)/
├── UrbanSound8K/UrbanSound8K/      the original dataset
├── data/                           generated audio (not in git)
│   ├── original/{train,val,test}/<class>/*.wav   + manifest.csv
│   └── synth/{train,val,test}/<class>/*.wav      + manifest.csv
└── SoundSense/                     this repo
    └── data/                       lists, scaling numbers, examples (in git)
```

## 3. What each file contains

### Lists: `data/trainSet1.csv`, `trainSet2.csv`, `trainSet3.csv`, `valSet.csv`, `testSet.csv`

One row per clip. The `file` path is relative to the outside `data/` folder (`../data` from the repo).

| Column | Meaning |
|---|---|
| `file` | e.g. `original/train/dog_bark/14468-3-0-3.wav` or `synth/test/dog_bark/..._snr10.wav` |
| `label` | class name (map to id with the table above) |
| `kind` | `original` or `mixed` |
| `condition` | `clean` (original), `mixed` (train/val mix, random SNR), `snr20` / `snr10` / `snr0` (test mixes) |
| `source_file` | the UrbanSound8K clip it came from |
| `source_fold` | UrbanSound8K fold |
| `place` | background place of a mix (empty for originals) |
| `snr_db` | SNR of a mix in dB (empty for originals) |

| List | Use | Contents |
|---|---|---|
| `trainSet1.csv` | training, experiment A | 3,993 originals (800 per class, children_playing 793) |
| `trainSet2.csv` | training, experiment B | same source clips: half as originals, half as mixes, never both |
| `trainSet3.csv` | training, experiment C | same source clips, all as mixes |
| `valSet.csv` | choosing epochs and settings | fold 9: 489 originals + 500 mixes |
| `testSet.csv` | final scores only | fold 10: 493 originals + 500 mixes each at 20, 10 and 0 dB |

All three training sets use exactly the same source clips, so they differ only in the type of data.

### Scaling numbers: `data/stats_trainSet#.json`

`mean` and `std` (in dB) of all log-mel values of that training set, plus the feature settings they were computed with. Scale every spectrogram as `(x - mean) / std`.

| Training set | Mean (dB) | Std (dB) |
|---|---|---|
| trainSet1 | -4.448 | 16.037 |
| trainSet2 | -3.279 | 14.331 |
| trainSet3 | -2.076 | 12.241 |

### Manifests: `../data/synth/manifest.csv` (copy in `data/manifest.csv`) and `../data/original/manifest.csv`

Full record of how each clip was made: source clip, fold, salience, crop start, position and length (in 16 kHz samples), background place, background start, SNR, seed. You only need them to trace a clip, for error analysis or the report.

### Code you will use

| File | What to use it for |
|---|---|
| `features.py` | `log_mel(clip)` turns a 2 s clip into the 64 × 201 log-mel spectrogram; `load_clip(path)`, `load_stats(path)`, `scale(x, mean, std)` |
| `adapter_for_app.py` | the app-side version: file or microphone audio → scaled spectrograms in 2 s windows every 1 s |

### Examples: `data/examples/`

One test event per class: the original clip, its background, and the mixes at 20, 10 and 0 dB. Listen to these first.

## 4. Rules you must follow

1. **Use only the lists.** Never pick files from the folders yourself, never reshuffle folds, never move clips between train, val and test.
2. **Test set only at the end.** Pick epochs, learning rate and every other setting on `valSet.csv`. Run `testSet.csv` once per finished model.
3. **One feature function.** Compute spectrograms with `features.log_mel`, or with the PyTorch call written at the top of `features.py`. If you use PyTorch, first check on a few clips that both give the same numbers (to about 1e-3 dB). The 80 dB floor (`top_db=80`) must be applied per clip: shape a batch as `(B, 1, 64, 201)`, not `(B, 64, 201)`.
4. **Scaling numbers from the training set you train on.** A model trained on trainSet2 uses `stats_trainSet2.json` for training, validation, test and in the app. Never compute them from val or test.
5. **Fair comparison.** The three models (trainSet1, 2, 3) must use the same architecture, seed, optimizer, learning rate, batch size, augmentation and stopping rule. Only the training list changes.
6. **Augmentation only during training, on the fly,** the same for all three sets. Time shift and SpecAugment (masking random time and frequency stripes) are useful. Gain (volume) augmentation does nothing useful here: every clip and every app window is peak-normalized, so the model never sees volume differences.
7. **Metric: macro-F1** (average of the per-class F1 scores) for every result.
8. **Report test results per condition:** `clean`, `snr20`, `snr10`, `snr0`, and overall. The main result is the score against noise level, with `clean` as the reference row.
9. **Save what the app needs with each model:** the weights, the name of the stats file, the class order above, and the feature settings (they are inside the stats file).

## 5. Loading the data in PyTorch

```python
import csv
from pathlib import Path

import numpy as np
import torch

import features as F

CLASSES = ["engine_idling", "air_conditioner", "dog_bark", "children_playing", "street_music"]
DATA = Path("../data")          # the outside data folder


class SoundSet(torch.utils.data.Dataset):
    def __init__(self, list_csv, stats_json):
        with open(list_csv, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.mean, self.std = F.load_stats(stats_json)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = F.scale(F.log_mel(F.load_clip(DATA / r["file"])), self.mean, self.std)
        return torch.from_numpy(x[None]), CLASSES.index(r["label"])   # (1, 64, 201), class id


train = SoundSet("data/trainSet2.csv", "data/stats_trainSet2.json")
val = SoundSet("data/valSet.csv", "data/stats_trainSet2.json")      # same stats as training
test = SoundSet("data/testSet.csv", "data/stats_trainSet2.json")
```

`log_mel` takes about 13 ms per clip. To make epochs faster, compute every spectrogram once and keep them in memory or in a `.npy` file. All 11,000 clips together take about 570 MB as float32.

For per-condition test scores, keep `r["condition"]` next to each prediction.

## 6. Facts to keep in mind

- **children_playing has 793 training clips**, not 800. Seven of its clips use a compressed format (MS-ADPCM) that cannot be read. Macro-F1 is not affected by this small imbalance.
- **Repeated events in val and test.** engine_idling has only 89 clips in fold 9 and 93 in fold 10, so 11 val mixes and 7 test mixes reuse an event with a different crop and background.
- **Each test event appears four times:** once clean and at 20, 10 and 0 dB. Compare conditions with each other; don't treat the 1,993 test rows as 1,993 independent events.
- **air_conditioner validation is weak.** Only 22 of its 100 fold-9 clips are salience 1 (foreground), so its validation score is noisy.
- **Expected confusions:** air_conditioner vs engine_idling (both steady low hums), and children_playing vs street_music. Check the confusion matrix for these first.
- **Little background audio.** All mixes come from about 5 minutes of the team's recordings (about 40 s per place for training, 8 s each for val and test). Test backgrounds are unheard stretches of the same rooms, so mixed-test scores are somewhat optimistic for brand-new rooms.
- **room has a steady fan hum** similar to air_conditioner. If air_conditioner errors cluster on room mixes, this is why.
- **No "other" class.** Any sound gets one of the 5 labels; the app relies on a low-confidence "not sure" rule and a silence gate.
- **Reference numbers:** a from-scratch CNN on all ten UrbanSound8K classes reached 73% accuracy, 79% with augmentation (Salamon & Bello, 2017). Five classes should be easier, so expect higher.
- **Baselines first:** majority class and MFCC + logistic regression, trained on each of the three sets. They give the first rows of the experiment log and catch pipeline bugs early.
