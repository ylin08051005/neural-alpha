from typing import Literal, Union

import numpy as np
import pandas as pd


def sharpe_ratio(ret: Union[pd.Series, np.ndarray, pd.DataFrame], rf: float) -> float:
    mean = ret.mean()
    std = ret.std()

    if std == 0:
        return 0

    return (mean - rf) / (std + 1e-8)


def get_next_n_day_sharpe(
    sequence: Union[pd.Series, pd.DataFrame],
    window: int,
    direction: Literal["pos", "neg"]
) -> pd.Series:
    sharpe_ratios = [None] * len(sequence)

    for i in range(len(sequence) - window):
        next_window_ret = sequence[i + 1 : i + window + 1]

        multiplier = 1 if direction == "pos" else -1
        sharpe_ratios[i] = multiplier * sharpe_ratio(next_window_ret, 0)

    return pd.Series(sharpe_ratios).fillna(0)


def get_next_n_day_cumret(
    sequence: Union[pd.Series, pd.DataFrame],
    window: int,
    direction: Literal["pos", "neg"]
) -> pd.Series:
    avg_rets = [None] * len(sequence)

    for i in range(len(sequence) - window):
        next_window_ret = sequence[i + 1 : i + window + 1]

        multiplier = 1 if direction == "pos" else -1

        if window > 1:
            avg_rets[i] = multiplier * next_window_ret.sum()
        else:
            avg_rets[i] = multiplier * next_window_ret.values[0]

    return pd.Series(avg_rets).fillna(0)
