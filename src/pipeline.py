from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import TrainAlphaDataset
from .utils.constants import feature_names, label_name
from .utils.utils import std_norm


class Pipeline:
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        shortest_sequence: int,
        train_scale: int,
        test_scale: int,
        prediction_window: int,
    ) -> None:
        """
        Args:
            data (Dict[str, pd.DataFrame]): Stock data
            shortest_sequence (int): Total number of days
            train_scale (int): Number of days to train on
            test_scale (int): Number of days to test on
            prediction_window (int): Number of days to predict & trade
        """
        self.data = data
        self.shortest_sequence = shortest_sequence
        self.train_scale = train_scale
        self.test_scale = test_scale
        self.prediction_window = prediction_window
        self.train_idxs, self.test_idxs = self.get_day_index()
        self.train_data, self.test_data = {}, {}
        self.feat_size = len(feature_names)

    def get_day_index(self):
        train_idxs, test_idxs = [], []

        for i in range(0, self.shortest_sequence, self.test_scale):
            if i + self.train_scale + self.test_scale > self.shortest_sequence:
                break

            train_idxs.append(i)
            test_idxs.append(i + self.train_scale)

        return train_idxs, test_idxs

    def prepare_data(self):
        for i, (train_idx, test_idx) in enumerate(zip(self.train_idxs, self.test_idxs)):
            self.train_data[i], self.test_data[i] = [], []

            for _, stock_data in self.data.items():
                self.train_data[i].append(
                    stock_data.iloc[train_idx:test_idx][
                        feature_names + label_name
                    ].to_numpy()
                )
                self.test_data[i].append(
                    stock_data.iloc[test_idx : test_idx + self.test_scale][
                        feature_names + label_name
                    ].to_numpy()
                )

            self.train_data[i] = np.transpose(np.array(self.train_data[i]), (1, 0, 2))
            self.test_data[i] = np.transpose(np.array(self.test_data[i]), (1, 0, 2))

    def run(
        self,
        model: Any,
        trainer: Any,
        model_path: str,
        stock_amount: int,
        loss_multiplier: float,
    ):
        self.prepare_data()

        for i in range(len(self.train_data)):
            train_feat = std_norm(self.train_data[i][:, :, :-1])
            train_label = self.train_data[i][:, :, -1]
            test_feat = std_norm(self.test_data[i][:, :, :-1])
            test_label = self.test_data[i][:, :, -1]
            train_dataset = TrainAlphaDataset(train_feat, train_label)
            test_dataset = TrainAlphaDataset(test_feat, test_label)
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            trainer.multi_asset_train(
                100,
                train_loader,
                optimizer,
                loss_multiplier=loss_multiplier,
                ndcg_k=stock_amount,
            )
            predictions = trainer.multi_asset_test(
                test_loader, model_path, self.prediction_window
            )

            return predictions
