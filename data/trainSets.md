# Training sets

| Set | What's in it | engine_idling | air_conditioner | dog_bark | children_playing | street_music | Total |
|---|---|---|---|---|---|---|---|
| `trainSet1.csv` | original only | 800 | 800 | 800 | 793 | 800 | 3,993 |
| `trainSet2.csv` | half original, half mixed | 400 + 400 | 400 + 400 | 400 + 400 | 397 + 396 | 400 + 400 | 3,993 |
| `trainSet3.csv` | mixed only | 800 | 800 | 800 | 793 | 800 | 3,993 |

- All three sets use the same source clips (UrbanSound8K folds 1–8), so they differ only in the type of data.
- In `trainSet2.csv` each source clip appears once, either as its original or as its mix, never both.
- The mixed clips are spread evenly over the 5 background places (house, kitchen, lecture, room, university).
- children_playing has 793 clips because 7 of its clips use a compressed format (MS-ADPCM) that cannot be read.
- Paths in the lists are relative to the `data` folder next to this repo, which holds `original/` (from `adapter.py`) and `synth/` (from `synthesizer.py`).
- Rebuild the lists with `python make_sets.py`.

## Validation and test sets

Shared by every model, so all results are comparable.

| List | Fold | Contents |
|---|---|---|
| `valSet.csv` | 9 | 489 originals (`clean`) + 500 mixes (`mixed`, random SNR 0–20 dB) |
| `testSet.csv` | 10 | 493 originals (`clean`) + 500 mixes each at `snr20`, `snr10`, `snr0` |

The `condition` column in every list (`clean`, `mixed`, `snr20`, `snr10`, `snr0`) lets results be reported per condition.

## Scaling numbers

`stats_trainSet#.json` holds the mean and standard deviation of the log-mel spectrograms of that training set (training clips only). Every spectrogram given to a model trained on that set, including in the app, is scaled as `(x - mean) / std`. Recompute with `python features.py`.

| Training set | Mean (dB) | Std (dB) |
|---|---|---|
| trainSet1 | -4.448 | 16.037 |
| trainSet2 | -3.279 | 14.331 |
| trainSet3 | -2.076 | 12.241 |
