"""Dataset for evaluation: one sample per available eye image.

Expected inputs
---------------
CSV file with one row per participant:
    sno                  participant ID
    prior_<disease>      0/1, empty/NA = unknown
    incident_<disease>   0/1, empty/NA = unknown
Label columns are optional; without them the script only writes predictions.

Image directory with 224x224 colour fundus photographs named
    <sno>_21015_<instance>_0.png   (left eye)
    <sno>_21016_<instance>_0.png   (right eye)
Instance 0 is used when available, otherwise instance 1.
"""
import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EYE_CODES = {"left": "21015", "right": "21016"}

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class FundusDiseaseDataset(Dataset):
    def __init__(self, csv_file, image_dir, disease_names, transform=TRANSFORM):
        self.transform = transform
        df = pd.read_csv(csv_file, dtype={"sno": str}, low_memory=False)
        if "sno" not in df.columns:
            raise ValueError(f"{csv_file} has no 'sno' (participant ID) column")

        # labels: [participants, diseases, 2] = (prior, incident), NaN when unknown
        labels = np.full((len(df), len(disease_names), 2), np.nan, dtype=np.float32)
        for d, name in enumerate(disease_names):
            for t, prefix in enumerate(("prior_", "incident_")):
                if prefix + name in df.columns:
                    labels[:, d, t] = pd.to_numeric(df[prefix + name], errors="coerce")
        # prior disease with missing incident status counts as a positive incident label
        fix = (labels[..., 0] == 1) & np.isnan(labels[..., 1])
        labels[..., 1][fix] = 1.0

        self.samples = []  # (image path, participant ID, eye, row index)
        for row, pid in enumerate(df["sno"]):
            for eye, code in EYE_CODES.items():
                for instance in (0, 1):
                    path = os.path.join(image_dir, f"{pid}_{code}_{instance}_0.png")
                    if os.path.exists(path):
                        self.samples.append((path, pid, eye, row))
                        break
        self.labels = labels
        if not self.samples:
            raise FileNotFoundError(
                f"No images found in {image_dir} for the participants in {csv_file} "
                "(expected <sno>_21015_0_0.png / <sno>_21016_0_0.png)")
        print(f"{csv_file}: {len(set(s[1] for s in self.samples))} participants, {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, pid, eye, row = self.samples[index]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, self.labels[row], pid
