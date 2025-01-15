from typing import Any

import numpy as np
import torch
from rich.progress import track
from torch.utils.data import DataLoader

from .dataset import TrainAlphaDataset
from .utils.utils import rolling


class Trainer:
    def __init__(self, model: Any, criterion: Any) -> None:
        self.model = model
        self.criterion = criterion
        self.buffer = None

    def train(self, n_epochs: int, train_loader: DataLoader, optimizer: Any) -> None:
        for epoch in track(range(n_epochs), description="Training..."):
            epoch_loss = 0
            self.model.train(True)

            for feature, label in train_loader:
                feature, label = feature.to(torch.float32).squeeze(0), label.to(
                    torch.float32
                )
                optimizer.zero_grad()
                y_pred = self.model(feature).transpose(0, 1)
                loss = self.criterion(y_pred, label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch} | Train Loss: {epoch_loss / len(train_loader)}")

    def test(self, test_features: torch.Tensor, test_label: torch.Tensor) -> float:
        with torch.no_grad():
            self.model.train(False)
            y_pred = self.model(test_features)
            corr = (
                -1
                * self.criterion(
                    y_pred.transpose(0, 1), test_label.transpose(0, 1)
                ).item()
            )

        return corr

    def trade_pipeline(
        self,
        threshold: float,
        init_buffer: torch.Tensor,
        full_test_feat: torch.Tensor,
        full_test_label: torch.Tensor,
        look_back_window: int,
        n_epochs: int,
        optimizer: Any,
    ) -> None:
        self.buffer = init_buffer

        for feats, label in zip(full_test_feat, full_test_label):
            label = label.unsqueeze(-1)
            corr = self.test(feats, label)

            if np.abs(corr) > threshold:
                print(f"Alpha still effective, continue trading | corr = {corr}")
                self.buffer = torch.concat(
                    [self.buffer[look_back_window:, :], torch.concat([feats, label], dim=1)],
                    dim=0
                )

            else:
                print(
                    f"Alpha is no longer effective, retrain model with new data | corr = {corr}"
                )
                print(f"Buffer size: {len(self.buffer)}")

                window_data = rolling(self.buffer, look_back_window)
                buffer_dataset = TrainAlphaDataset(window_data)
                buffer_loader = DataLoader(buffer_dataset, batch_size=1, shuffle=True)
                self.model._reset_parameters()
                self.train(n_epochs, buffer_loader, optimizer)
