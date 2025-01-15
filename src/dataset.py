from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .utils.metrics import *


class TrainAlphaDataset(Dataset):

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.features = data[:, :, :-1]
        self.labels = data[:, :, -1]
        self._feature_normalization()

    def _feature_normalization(self):
        self.features = (
            self.features - self.features.mean(axis=0)
        ) / self.features.std(axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def label_map(label_type: Literal["avgret", "sharpe"]) -> Callable:
    if label_type == "avgret":
        return get_next_n_day_avgret
    elif label_type == "sharpe":
        return get_next_n_day_sharpe


def preprocess(
    data: Any, label_type: Literal["avgret", "sharpe"], window: int
) -> pd.DataFrame:
    data["ret"] = data["收盤價"].pct_change().fillna(0)
    data["date"] = pd.to_datetime(data["date"])
    labeler = label_map(label_type)
    data["label"] = labeler(data["ret"], window=window)

    return data
