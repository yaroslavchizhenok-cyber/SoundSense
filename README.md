# SoundSense

SoundSense recognizes everyday urban sounds from an uploaded recording or a live microphone. It classifies each 2-second window of audio into one of five classes:

**engine idling · air conditioner · dog bark · children playing · street music**

Team project for Artificial Intelligence [SOC4040], IUT, Fall 2026.

## Approach

1. **Data.** Events come from the public [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset. The team recorded its own background noise in five places (house, kitchen, lecture, room, university) and mixed the events over it at controlled noise levels. This gives realistic noisy clips and lets us measure how accuracy falls as noise rises.
2. **Preprocessing.** Every clip becomes 2 s, 16 kHz, mono, peak-normalized audio, then a 64 × 201 log-mel spectrogram, scaled with numbers from the training set.
3. **Model.** A convolutional neural network written from scratch in PyTorch.
4. **Experiment.** The same model is trained on three training sets: original clips only, half original and half mixed, and mixed clips only. Each is scored with macro-F1 on clean test clips and on test mixes at 20, 10 and 0 dB.
5. **App.** A local web app that loads the trained model and classifies uploaded files or live microphone audio.

## Repository

| Path | Contents |
|---|---|
| `synthesizer.py` | Mixes UrbanSound8K events over the team's background recordings → mixed clips + manifest |
| `adapter.py` | Converts original UrbanSound8K clips to the same format as the mixed clips |
| `make_sets.py` | Builds the training, validation and test lists in `data/` |
| `features.py` | The log-mel spectrogram function used everywhere; computes the scaling numbers |
| `adapter_for_app.py` | App-side preprocessing: files or microphone audio → model input, with input checks and a silence gate |
| `noise/` | The team's five background recordings (`.m4a` originals and 16 kHz `.wav` copies) |
| `data/` | Lists (`trainSet1-3.csv`, `valSet.csv`, `testSet.csv`), scaling numbers (`stats_trainSet#.json`), mix manifest, example clips |
| `data_manual.md` | **Read this before training a model:** what each data file contains and the rules for the experiments |
| `app/` | Web app (`backend/`, `frontend/`); not started yet |
| `SoundSense_context.md` | Planning record: grading, plan, roles, decisions |

## Getting started

Requirements: Python 3.12 and NumPy. ffmpeg is optional (only for non-WAV audio). PyTorch is needed for the model.

1. Download UrbanSound8K ([Kaggle: chrisfilo/urbansound8k](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)) and extract it next to this repo, so that `../UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv` exists.
2. From inside this folder, generate the audio:

   ```
   python synthesizer.py     # mixed clips    -> ../data/synth/     (~2 min)
   python adapter.py         # original clips -> ../data/original/  (~1 min)
   ```

3. The lists and scaling numbers in `data/` are already committed. Rebuild them only if the audio changes: `python make_sets.py`, then `python features.py`.

Generated audio (about 700 MB) is written outside the repo and is never committed; the scripts are deterministic, so everyone gets identical files. Each script has more options, listed by `python <script>.py --help`.

## Data at a glance

| Split | UrbanSound8K folds | Original clips | Mixed clips |
|---|---|---|---|
| Train | 1–8 | 3,993 (800 per class; children_playing 793) | 4,000 (0–20 dB) |
| Validation | 9 | 489 | 500 (0–20 dB) |
| Test | 10 | 493 | 1,500 (each event at 20, 10 and 0 dB) |

- **Folds are never reshuffled.** A mixed clip inherits the fold of its source clip.
- **No leakage.** Each background recording is cut into 4 s blocks dealt about 70/15/15 to train/val/test. No source clip and no stretch of background audio is in two splits.
- **Even use.** Each place is used equally often within every split and class, and training SNRs cover 0–20 dB evenly.
- **Same piece in both forms.** An original clip and its mix contain the same 2 s piece of the event at the same position, so the two differ only by the background.

## Useful facts for development

- **Feature settings:** 16 kHz, 25 ms Hann window (400 samples), 10 ms hop (160), FFT 400, centred frames, 64 HTK-mel bands from 50 to 8,000 Hz, power in dB with an 80 dB floor below each clip's maximum. The matching PyTorch call is in `features.py`.
- **Scaling numbers depend on the training set.** A model must use the `stats_trainSet#.json` of the set it was trained on, everywhere including the app.
- **Unreadable clips:** 7 children_playing clips (Freesound ID 36429) use MS-ADPCM compression and are skipped by every script.
- **Few clips:** engine_idling has 89 clips in fold 9 and 93 in fold 10; some of its val and test mixes reuse an event.
- **Weak foregrounds:** only 22 of air_conditioner's 100 fold-9 clips are foreground (salience 1).
- **The background recordings are short:** about 1 minute each, 48 kHz AAC, effectively mono. More minutes from the same places would make the mixes more varied.
- **The room recording has a steady fan hum** that resembles air_conditioner.
- **App silence gate:** windows quieter than -50 dBFS are not classified. Tune this on the demo laptop's microphone.
- **No "other" class:** every non-silent sound gets one of the five labels, so the app needs a low-confidence "not sure" answer.
- **Browser microphone audio** usually arrives as `.webm`, which needs ffmpeg on the server; alternatively send raw samples to `adapter_for_app.prepare_samples`.
- **Line endings:** `.gitattributes` keeps text files LF on every OS and marks audio as binary.

## Status

- [x] Background recordings
- [x] Mixing and original-clip conversion
- [x] Training, validation and test lists; scaling numbers
- [x] Preprocessing for the app
- [ ] Baselines (majority class, MFCC + logistic regression)
- [ ] CNN, trained on the three training sets
- [ ] Evaluation, confusion matrix, error analysis
- [ ] Web app (upload and live microphone)
- [ ] Report and presentation

## Credits and licence of reused material

- **UrbanSound8K:** J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", ACM Multimedia 2014. Licence: CC BY-NC 3.0 (non-commercial use only). Clips are excerpts of Freesound recordings; see `FREESOUNDCREDITS.txt` in the dataset.
- **Background recordings and all code:** made by the team. All mixed clips are synthesized and labelled as such.
- **Benchmark:** J. Salamon and J. P. Bello, "Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification", IEEE Signal Processing Letters, 2017.
