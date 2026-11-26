# Experiment log 

Every model trained for SoundSense, with its settings and scores. One row per run. All scores are
**macro-F1**. Validation = `valSet.csv` (fold 9), test = `testSet.csv` (fold 10), both shared by every run.

Regenerate the table at any time with `python summarize.py` (writes `runs/results.md` / `runs/results.csv`).
Numbers for the analysis below come from `python analysis.py` (writes `runs/analysis/`).

All CNNs were trained  with the same settings (see `train.py`, `model.py`); runs differ only
in model, training set and augmentation (`+ aug`). For every CNN the best epoch was chosen on validation
macro-F1, never on test, so the test scores below are unbiased.

## Runs

| # | Run | Model | Train set | Best epoch / ran | Val | Test clean | 20 dB | 10 dB | 0 dB | Test overall | Who |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | `baseline_majority_set1..3` | majority class | 1 / 2 / 3 | – | 0.064 | 0.064 | 0.067 | 0.067 | 0.067 | 0.066 | me |
| B1 | `baseline_mfcc_lr_set1` | MFCC + log. reg. (C=0.1) | 1 | – | 0.590 | 0.675 | 0.667 | 0.629 | 0.527 | 0.625 | me |
| B2 | `baseline_mfcc_lr_set2` | MFCC + log. reg. (C=10) | 2 | – | 0.586 | 0.674 | 0.675 | 0.642 | 0.543 | 0.633 | me |
| B3 | `baseline_mfcc_lr_set3` | MFCC + log. reg. (C=10) | 3 | – | 0.566 | 0.661 | 0.666 | 0.648 | 0.559 | 0.634 | me |
| M1 | `small_set2_seed42` | small CNN | 2 | 49 / 50 | 0.613 | 0.668 | 0.686 | 0.657 | 0.513 | 0.631 | Builder of the models (test: me) |
| M2 | `deep_set2_seed42` | deep CNN | 2 | 8 / 18 | 0.710 | 0.770 | 0.764 | 0.746 | 0.618 | 0.725 | Builder of the models (test: me) |
| M3 | `deep_aug_set2_seed42` | deep CNN + aug | 2 | 12 / 22 | 0.729 | 0.780 | 0.787 | 0.757 | 0.675 | 0.750 | Builder of the models |
| K1 | `deep_aug_set1_seed42` | deep CNN + aug | 1 (original only) | 13 / 23 | 0.720 | 0.705 | 0.690 | 0.673 | 0.521 | 0.650 | Builder of the models |
| K2 | = M3 | deep CNN + aug | 2 (half / half) | 12 / 22 | 0.729 | 0.780 | 0.787 | 0.757 | 0.675 | 0.750 | Builder of the models |
| **K3** | **`deep_aug_set3_seed42`** | **deep CNN + aug** | **3 (mixed only)** | 13 / 23 | **0.745** | **0.798** | **0.800** | **0.788** | **0.717** | **0.776** | Builder of the models |

Training sets: 1 = original clips only, 2 = half original / half mixed, 3 = mixed only. All three use the
same 3,993 source events from folds 1–8, so they differ only in the kind of audio.

**Final model: K3 `deep_aug_set3_seed42`**, chosen because it has the highest validation macro-F1
(0.745). The test set was not used to choose it. It is also best on test in every condition.

## Log entries

1. *Data.* All my work uses only the audio and spectrogram cache that the Builder of the models trained
   on (`data.zip` / `cache.zip`, unpacked next to the repo, unmodified) and the lists, stats and manifest
   from the branch of the Builder of the data (`origin/person_B`). The `data/synth/manifest.csv` used for
   training is byte-identical to the `data/manifest.csv` committed by the Builder of the data, and every
   file named in the five lists exists in that audio.
2. *Test results of the Builder of the models reproduced* by re-running `evaluate.py` on copies of those
   checkpoints with the same data (CPU): K1, K2, K3 each differ in 1 of 1,993 predictions, all near-ties
   (confidence 0.30–0.49), macro-F1 within 0.002 (the original run used a GPU).
3. *Test-evaluated M1 and M2* (the Builder of the models had only validation scores), so every row is
   scored on the same test data.
4. *Baselines B0–B3* added (`baselines.py`), each trained on all three training sets.
   MFCC: 20 coefficients from a DCT of `features.log_mel`, mean + std over time (40 features),
   standardized, logistic regression with C ∈ {0.01, 0.1, 1, 10} chosen on validation.
5. *Final model picked* (K3) and *error analysis* done (`analysis.py`, see `RESULTS.md`).

## Issues found / open questions


- Validation macro-F1 jumps a lot between epochs (K3: 0.39–0.74; best 0.744 at epoch 13, next epoch 0.553), so the "best epoch" is partly luck.
- M1 (small CNN) was still improving at epoch 49 of 50.
- Only M1–M3 compare models, and only on set 2; the training-set comparison is done only for deep + aug.
- MFCC baseline picked C = 10 (edge of the grid) for sets 2 and 3.
