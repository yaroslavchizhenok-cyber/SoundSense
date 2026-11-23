import csv
from pathlib import Path

import numpy as np
import torch

import features as F

# Our five sound classes
CLASSES = [
    "engine_idling",
    "air_conditioner",
    "dog_bark",
    "children_playing",
    "street_music",
]

# Audio files are stored outside the GitHub repository
DATA = Path("../data")


class SoundSet(torch.utils.data.Dataset):
    def __init__(self, list_csv, stats_json):
        with open(list_csv, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        self.mean, self.std = F.load_stats(stats_json)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]

        audio = F.load_clip(DATA / r["file"])
        spectrogram = F.log_mel(audio)
        scaled = F.scale(spectrogram, self.mean, self.std)

        x = torch.from_numpy(scaled[None])
        y = CLASSES.index(r["label"])

        return x, y


# Load the first training dataset
train = SoundSet(
    "data/trainSet1.csv",
    "data/stats_trainSet1.json",
)

print("Number of training clips:", len(train))

# Inspect the first example
x, y = train[0]

print("Input shape:", x.shape)
print("Class ID:", y)
print("Class name:", CLASSES[y])
print("Data type:", x.dtype)