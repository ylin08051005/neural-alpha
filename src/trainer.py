from typing import Any

import numpy as np
import torch
from rich import print
from rich.progress import track
from torch.utils.data import DataLoader

from .dataset import TrainAlphaDataset
from .utils.utils import EarlyStopping
from .model.eval import ndcg


class Trainer:
    def __init__(
        self, model: Any, model_type: str, criterion: Any, device: str, model_path: str
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.criterion = criterion
        self.device = device
        self.model_path = model_path
        self.buffer = None
        self.patience = 10
        self.verbose = True
        self.wandb = None

    def multi_asset_train(
        self, n_epochs: int, train_loader: DataLoader, optimizer: Any
    ) -> None:
        early_stopping = EarlyStopping(
            model_path=self.model_path, patience=self.patience, verbose=self.verbose
        )
        for epoch in track(range(n_epochs)):
            epoch_loss = 0
            self.model.train(True)

            for _, (feature, label) in enumerate(train_loader):
                if self.model_type == "attention":
                    feature, label = feature.to(torch.float32), label.to(torch.float32)
                else:
                    feature, label = feature.to(torch.float32).squeeze(0), label.to(
                        torch.float32
                    ).squeeze(0)

                feature, label = feature.to(self.device), label.to(self.device)
                y_pred = self.model(feature)
                loss = self.criterion(y_pred, label)

                if self.wandb is not None:
                    self.wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": loss.item(),
                            "label_corr": -1 * loss.item(),
                        }
                    )

                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()

            early_stopping(epoch_loss / len(train_loader), self.model)

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Train Loss: {epoch_loss / len(train_loader)}")

            if early_stopping.early_stop:
                print("Early stopping")
                break

    def multi_asset_test(
        self, test_features: torch.Tensor, test_label: torch.Tensor, future_window: int
    ) -> list:
        y_preds = []

        for i, (test_features, test_label) in track(enumerate(zip(test_features, test_label))):
            if i % future_window == 0:
                if self.model_type == "attention":
                    test_features, test_label = (
                        torch.tensor(test_features, dtype=torch.float32).unsqueeze(0),
                        torch.tensor(test_label, dtype=torch.float32).unsqueeze(0),
                    )
                else:
                    test_features, test_label = (
                        torch.tensor(test_features, dtype=torch.float32),
                        torch.tensor(test_label, dtype=torch.float32),
                    )
                with torch.no_grad():
                    self.model.train(False)
                    y_pred = self.model(test_features)

                y_preds.append(y_pred[0].cpu().numpy())

        return y_preds

    def trade_pipeline(
        self,
        threshold: float,
        buffer_feat: np.ndarray,
        buffer_label: np.ndarray,
        full_test_feat: np.ndarray,
        full_test_label: np.ndarray,
        look_back_window: int,
        n_epochs: int,
        optimizer: Any,
    ) -> None:
        """
        buffer_feat (np.ndarray): original training dataset (n_days - look_back_window - 1, n_stock, n_feats * look_back_window)
        buffer_label (np.ndarray): original training label (n_days - look_back_window - 1, n_stock, 1)
        full_test_feat (np.ndarray): rolled test dataset (n_days - look_back_window - 1, n_stock, n_feats * look_back_window)
        full_test_label (np.ndarray): rolled test label (n_days - look_back_window - 1, n_stock, 1)
        """
        for feats, label in zip(full_test_feat, full_test_label):
            corr = self.multi_asset_test(feats, label, 5)

            if np.abs(corr) > threshold:
                print(f"Alpha still effective, continue trading | corr = {corr}")
                buffer_feat = np.concatenate(
                    [buffer_feat[1:], np.expand_dims(feats, axis=0)],
                    axis=0,
                )
                buffer_label = np.concatenate(
                    [buffer_label[1:], np.expand_dims(label, axis=0)],
                    axis=0,
                )

            else:
                print(
                    f"Alpha is no longer effective, retrain model with new data | corr = {corr}"
                )
                print(f"Buffer size: {len(buffer_feat)}")

                buffer_dataset = TrainAlphaDataset(buffer_feat, buffer_label)
                buffer_loader = DataLoader(buffer_dataset, batch_size=1, shuffle=True)
                self.model._reset_parameters()
                self.multi_asset_train(n_epochs, buffer_loader, optimizer)


class MultiAlphaTrainer:
    def __init__(
        self,
        model: Any,
        model_type: str,
        label_loss: Any,
        pool_loss: Any,
        device: str,
        model_path: str,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.label_loss = label_loss
        self.pool_loss = pool_loss
        self.device = device
        self.model_path = model_path
        self.buffer = None
        self.patience = 10
        self.verbose = True
        self.wandb = None

    def multi_asset_train(
        self,
        n_epochs: int,
        train_loader: DataLoader,
        optimizer: Any,
        loss_multiplier: float = 1.0,
        ndcg_k: int = 20,
    ) -> None:
        early_stopping = EarlyStopping(
            model_path=self.model_path, patience=self.patience, verbose=self.verbose
        )
        for epoch in track(range(n_epochs)):
            epoch_loss = 0
            epoch_pool_loss, epoch_label_loss = 0, 0
            self.model.train(True)

            for _, (feature, label) in enumerate(train_loader):
                if self.model_type == "attention":
                    feature, label = feature.to(torch.float32), label.to(torch.float32)
                else:
                    feature, label = feature.to(torch.float32).squeeze(0), label.to(
                        torch.float32
                    ).squeeze(0)

                feature, label = feature.to(self.device), label.to(self.device)
                y_pred = self.model(feature)

                label_losses, mutual_losses = torch.tensor(0.0), torch.tensor(0.0)

                for i in range(y_pred.shape[-1]):
                    label_loss = self.label_loss(y_pred[:, :, i], label)
                    label_losses += label_loss

                label_losses /= y_pred.shape[-1]

                for i in range(y_pred.shape[-1]):
                    for j in range(i + 1, y_pred.shape[-1]):
                        mutual_loss = self.pool_loss(y_pred[:, i], y_pred[:, j])
                        mutual_losses += mutual_loss

                mutual_losses /= y_pred.shape[-1] * (y_pred.shape[-1] - 1) / 2

                loss = label_losses + loss_multiplier * mutual_losses
                top_ndcg = ndcg(y_pred, label, ndcg_k)

                # if self.wandb is not None:
                #     self.wandb.log(
                #         {
                #             "epoch": epoch,
                #             "epoch_loss": epoch_loss,
                #             "train_loss": loss.item(),
                #             "label_loss": label_losses.item(),
                #             "pool_loss": mutual_losses.item(),
                #             "top_ndcg": top_ndcg,
                #         }
                #     )

                loss.backward()
                epoch_loss += loss.item()
                epoch_pool_loss += mutual_losses.item()
                epoch_label_loss += label_losses.item()
                optimizer.step()
                optimizer.zero_grad()

            if self.wandb is not None:
                self.wandb.log(
                    {
                        "epoch": epoch,
                        "epoch_loss": epoch_loss / len(train_loader),
                        "pool_loss": epoch_pool_loss / len(train_loader),
                        "label_loss": epoch_label_loss / len(train_loader),
                    }
                )

            early_stopping(epoch_loss / len(train_loader), self.model)

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Train Loss: {epoch_loss / len(train_loader)}")

            if early_stopping.early_stop:
                print("Early stopping")
                break

    def multi_asset_test(
        self, test_features: torch.Tensor, test_label: torch.Tensor, future_window: int
    ) -> list:
        y_preds = []

        for i, (test_features, test_label) in track(enumerate(zip(test_features, test_label))):
            if i % future_window == 0:
                if self.model_type == "attention":
                    test_features, test_label = (
                        torch.tensor(test_features, dtype=torch.float32).unsqueeze(0),
                        torch.tensor(test_label, dtype=torch.float32).unsqueeze(0),
                    )
                else:
                    test_features, test_label = (
                        torch.tensor(test_features, dtype=torch.float32),
                        torch.tensor(test_label, dtype=torch.float32),
                    )
                with torch.no_grad():
                    self.model.train(False)
                    y_pred = self.model(test_features)

                y_preds.append(y_pred[0].cpu().numpy())

        return y_preds

    def trade_pipeline(self, *args):
        pass
