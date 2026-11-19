# SoundSense: chat context (11–17 Nov 2026)

Compressed record of the planning chat. Where decisions changed, only the latest one is listed as current.

## Project

- **What:** SoundSense, real-time sound recognition from microphone input or uploaded recordings.
- **Course:** Artificial Intelligence [SOC4040], IUT, Fall 2026, Dr. Oybek Eraliev. Team project graded out of 100 by presentation and live demo.
- **Team:** 5 final-year CS students, labelled A to E (names not assigned yet).
- **Deadline:** 3 weeks from 11 Nov 2026. Presentation on about 2 Dec is an assumption, not confirmed.
- **Working preferences:** brief answers; create files only when asked.

## Guideline points that drive the plan

- **Grading:** Problem 10, Methodology 25, Dataset/Experiments/Results 25, Deployment and demo 10, Creativity 10, individual Q&A 20 (4 questions per member).
- **Valued most:** data collected or created by the team, a pipeline and model built from scratch, several experiments compared with one metric on the same test data, a stated reason for the final model.
- **Required:** local demo with a UI that loads the saved model, README to run it, steady GitHub history from every member, before/after preprocessing examples, disclosure of everything reused.
- **Submission:** project brief in the team's Discord space, code on GitHub, report on eClass, 8-slide presentation structure.
- **Compute:** IUT AI Lab access must be requested early through the education assistant.

## Current plan

### Data

- **Dataset:** UrbanSound8K only (8,732 clips up to 4 s, 10 predefined folds, `salience` column). On Kaggle: `chrisfilo/urbansound8k`.
- **Classes:** engine idling, air conditioner, dog bark, children playing, street music (1,000 clips each).
- **Left out:** siren, drilling, jackhammer, car horn, gun shot.
- **Split:** folds 1–8 train, fold 9 validation, fold 10 test. Never reshuffle. A synthesized clip inherits the fold of its source clip.
- **Backgrounds:** 5 recordings made by the team in `SoundSense/noise/` (house, kitchen, lecture, room, university), about 1 minute each. Every recording is used in every split: each one is cut into 4 s blocks dealt about 70/15/15 to train/validation/test, so no stretch of audio is shared between splits. Each place is used equally often within every split and class.

Rows per class (synthesized rows are stored, not generated on the fly):

| Split | Original | Synthesized (original + own background) |
|---|---|---|
| Train (folds 1–8) | 800 | 800 |
| Validation (fold 9) | 100 | 100 |
| Test (fold 10) | 100 | 300 (each event at 20, 10 and 0 dB) |

Total stored mixes: 6,000 files, under 400 MB.

### Mixing and augmentation

- **Mixer:** `synthesizer.py`, written in NumPy, not Scaper. Places the event at a random point in a 2 s window over a random background crop.
- **Noise level:** training and validation mixes spread evenly over 0–20 dB signal-to-noise; test mixes at fixed 20, 10 and 0 dB.
- **Foregrounds:** `salience` 1 clips are used first. Air conditioner has only 22 salience-1 clips in fold 9.
- **Rules:** same backgrounds for every class, peak-normalize after mixing, fixed random seed.
- **Manifest:** one CSV row per stored clip with source clip, fold, background file and place, noise level, seed.
- **On the fly during training:** gain ±6 dB, time shift, SpecAugment.
- **Optional experiments:** pitch shift ±2 semitones, time stretch 0.9–1.1.

### Preprocessing

- **Audio:** 16 kHz, mono, peak-normalized, 2 s windows with a 1 s hop; short clips zero-padded in validation and test.
- **Features:** log-mel, 64 bands, 25 ms window, 10 ms hop, 50–8,000 Hz, giving a 64 × 201 input.
- **Scaling:** mean and standard deviation from the training set only.
- **Shared code:** one preprocessing function used by both training and the app. Live mode adds a silence gate.

### Model and experiments

- **Model:** CNN written by hand in PyTorch. Small CNN first, then a deeper one with batch norm.
- **Baselines:** majority class; MFCC + logistic regression.
- **Metric:** macro-F1 for every row of the experiment log.
- **Key comparison:** three training sets: original clips only, original + mixed half and half, mixed only. Each is scored on clear clips and on mixed clips.
- **Main result:** accuracy against noise level (20, 10, 0 dB), with plain fold 10 as the reference row.
- **Benchmarks:** from-scratch CNN on UrbanSound8K's ten classes reached 73%, and 79% with augmentation (Salamon & Bello). ESC-50 baseline CNN reached 64.5% at 40 clips per class.

### Dataset quality checks

1. **Listening:** two teammates blind-label 20 random clips per class; pass at 90% agreement with the true label.
2. **File check:** script confirms 2 s, 16 kHz, mono, not silent, not clipped, balanced classes.
3. **Leakage:** no source clip and no stretch of background audio shared between train, validation and test.
4. **Behaviour:** small model's accuracy should fall steadily as noise rises; near-random or near-perfect means a fault.
5. **Reality:** record 10–15 real clips per class on a phone and compare the score with the created test set.

### App

- **Features:** upload a file or use the live microphone; loads the saved model.
- **Edge cases:** silence, unknown sounds (low confidence shows "not sure"), wrong file types, long files.
- **Demo prep:** test on the presenting laptop and a second laptop from the README; keep a backup screen recording.

### Honest wording for the report

- **Test set:** "created by the team from public events and our own background recordings", not "collected".
- **Disclosure:** label every synthesized clip and list everything reused.

## Roles

| Person | Role | Week 1 | Week 2 | Week 3 |
|---|---|---|---|---|
| A | Data creation | Fix classes, post brief in Discord, record backgrounds, write mixer | Generate mixed set, run dataset quality checks, before/after examples | Dataset part of report |
| B | Data pipeline | Download data, set fold split, build preprocessing | Augmentation, leakage check | Clean repo, runnable README, pipeline part |
| C | Model | First small CNN and training loop | Deeper, augmented and synthetic-data versions | Freeze model, methodology part, architecture diagram |
| D | Experiments | Experiment log, baselines, score first model | Evaluate all checkpoints, pick final model | Confusion matrix, error analysis, results part |
| E | Application | App that predicts on an uploaded file | Live microphone mode, input handling | Demo tests on two laptops, backup video |

All five: own slide, two rehearsals with the live demo, mock Q&A, submit report and code. Jobs within a week run in order A to E, each using the previous output.

## Dropped ideas

- **Rain class:** dropped, which removed ESC-50, FSD50K and procedural rain.
- **Own recorded event dataset:** replaced by mixing public events with own backgrounds (only the 10–15 clip reality check remains).
- **Replayed and downloaded internet audio:** not needed.

## Open items

- **Not yet decided:** which member is A to E, the presentation date, and whether the AI Lab request was sent.

## Files

- Team Project Guidelines PDF (IUT AI, Fall 2026).

## Sources

- [UrbanSound8K on Kaggle](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)
- [UrbanSound8K dataset card](https://huggingface.co/datasets/danavery/urbansound8K/blob/main/README.md)
- [Salamon & Bello, CNNs and data augmentation for environmental sound classification](https://ar5iv.labs.arxiv.org/html/1608.04363)
- [ESC-50 repository](https://github.com/karolpiczak/ESC-50)
