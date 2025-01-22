import os
import random
from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np
import torch

from .constants import feature_names, label_name


class EarlyStopping:
    def __init__(self, model_path: str, patience: int = 10, verbose: bool = True) -> None:
        self.patience = patience
        self.verbose = verbose
        self.model_path = model_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = float("inf")

    def __call__(self, loss: float, model: Any) -> None:
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)

        elif score < self.best_score:
            self.counter += 1

            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss: float, model: Any) -> None:
        if self.verbose:
            print(
                f"Training loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ..."
            )

        torch.save(model.state_dict(), f"model/{self.model_path}")
        self.loss_min = loss


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def std_norm(data: np.ndarray) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    return (data - data.mean(axis=0, keepdims=True)) / data.std(axis=0, keepdims=True)


def rolling(data: np.ndarray, look_back_window: int) -> Tuple[np.ndarray, np.ndarray]:
    train_window_feat, train_window_label = [], []

    for i in range(data.shape[0] - look_back_window - 1):
        train_window_feat.append(
            np.concatenate(std_norm(data[i : i + look_back_window, :, :-1]), axis=1)
        )
        train_window_label.append(
            np.concatenate(
                np.expand_dims(
                    data[i + look_back_window : i + look_back_window + 1, :, -1],
                    axis=-1,
                ),
                axis=1,
            )
        )

    return np.array(train_window_feat), np.array(train_window_label)


def dict2mat(
    data: Dict[str, pd.DataFrame], train_scale: int, shortest_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    full_tr_list, full_ts_list = [], []

    for _, stock_data in data.items():
        full_tr_list.append(
            stock_data.iloc[:train_scale][feature_names + label_name].to_numpy()
        )
        full_ts_list.append(
            stock_data.iloc[train_scale:shortest_len][
                feature_names + label_name
            ].to_numpy()
        )

    return np.array(full_tr_list), np.array(full_ts_list)
