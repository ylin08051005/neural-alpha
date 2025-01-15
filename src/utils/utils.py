import random

import numpy as np
import pandas as pd
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rolling(data: torch.Tensor, look_back_window: int) -> np.ndarray:
    window_data = []

    for i in range(len(data) - look_back_window):
        window_data.append(data[i : i + look_back_window])

    return np.array(window_data)