"""
PyTorch Dataset Loader for Indonesian Hoax News Detection
"""

import torch
from torch.utils.data import Dataset
import pandas as pd

class IndonesianHoaxDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128, text_column: str = "full_content"):
        self.texts = df[text_column].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

