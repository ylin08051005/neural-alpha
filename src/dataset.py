from collections.abc import Callable
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
from rich.progress import track
from torch.utils.data import Dataset

from .utils.metrics import (
    get_next_n_day_cumret,
    get_next_n_day_sharpe,
)


class TrainAlphaDataset(Dataset):
    """
    stock dataset for alpha mining

    Args:
        self.data (np.ndarray): stock data with dim (n_days, n_stocks, feats + labels)
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        features (np.ndarray): (train_len - look_back_window, n_stocks, n_feats * look_back_window)
        labels (np.ndarray): (train_len - look_back_window, n_stocks, n_labels * future_window)
        """
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.features[idx], self.labels[idx]


def label_map(label_type: Literal["cumret", "sharpe"]) -> Callable:
    if label_type == "cumret":
        return get_next_n_day_cumret
    elif label_type == "sharpe":
        return get_next_n_day_sharpe


def preprocess(
    data: Dict[str, pd.DataFrame],
    label_type: Literal["cumret", "sharpe"],
    window: int,
    direction: Literal["pos", "neg"],
) -> Dict[str, pd.DataFrame]:
    for stock_id, stock_data in track(
        data.items(), description="Preprocessing data..."
    ):
        stock_data["ret"] = stock_data["收盤價"].pct_change().fillna(0)
        stock_data["date"] = pd.to_datetime(stock_data["date"])
        labeler = label_map(label_type)
        stock_data["label"] = labeler(stock_data["ret"], window=window, direction=direction)
        data[stock_id] = stock_data

    return data
