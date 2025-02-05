from typing import Dict

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from .dataset import TrainAlphaDataset
from .model.loss import ICLoss, InverseICLoss
from .model.nn_model import AlphaSelfAttention
from .trainer import MultiAlphaTrainer
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
        device: str,
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
        self.device = device
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
        wandb_track: bool,
        n_epochs: int,
        stock_amount: int,
        train_batch: int,
        loss_multiplier: float,
        wandb_config: DictConfig,
    ):
        self.prepare_data()

        if wandb_track:
            wandb.init(
                project=wandb_config.project_name,
                name=wandb_config.run_name,
                config={
                    "learning_rate": wandb_config.lr,
                    "architecture": "self_attention",
                    "batch_size": wandb_config.batch_size,
                    "epochs": wandb_config.epochs,
                    "stock_amount": stock_amount,
                    "train_batch": train_batch,
                    "prediction_window": self.prediction_window,
                },
            )

        full_predictions = []

        for i in range(len(self.train_data)):
            if wandb_track:
                wandb.define_metric(f"iter_{i}/train/*", step_metric="epoch")
                wandb.define_metric(f"iter_{i}/test/*", step_metric="epoch")

            model = AlphaSelfAttention(
                input_dim=self.feat_size,
                embed_dim=128,
                num_heads=1,
                dropout=0.1,
                kdim=None,
                vdim=None,
                final_output_dim=10,
                device=self.device,
                dtype=torch.float32,
                value_weight_type="vanilla",
            ).to(self.device)

            if wandb_track:
                wandb.watch(model, log="all", log_freq=10)

            label_ic = ICLoss(ic_type="spearman")
            pool_ic = InverseICLoss(ic_type="spearman")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            model_path = f"{self.prediction_window}d_cumret_10_alphas_eta_{loss_multiplier}_iter_{i}.pt"
            trainer = MultiAlphaTrainer(
                model,
                model_type="attention",
                label_loss=label_ic,
                pool_loss=pool_ic,
                device=self.device,
                model_path=model_path,
            )

            if wandb_track:
                trainer.wandb = wandb
                trainer.current_iteration = i

            train_feat = std_norm(self.train_data[i][:, :, :-1])
            train_label = self.train_data[i][:, :, -1]
            test_feat = std_norm(self.test_data[i][:, :, :-1])
            test_label = self.test_data[i][:, :, -1]
            train_dataset = TrainAlphaDataset(train_feat, train_label)
            test_dataset = TrainAlphaDataset(test_feat, test_label)
            train_loader = DataLoader(
                train_dataset, batch_size=train_batch, shuffle=True
            )
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            train_metrics = trainer.multi_asset_train(
                n_epochs,
                train_loader,
                optimizer,
                loss_multiplier=loss_multiplier,
                ndcg_k=stock_amount,
            )
            predictions = trainer.multi_asset_test(
                test_loader, model_path, self.prediction_window
            )
            full_predictions.append(predictions)

            if wandb_track:
                wandb.log(
                    {
                        f"iter_{i}/summary": wandb.plot.line_series(
                            xs=list(range(n_epochs)),
                            ys=[train_metrics["train_losses"]],
                            keys=["Train Loss"],
                            title=f"Iteration {i} Loss Curves",
                            xname="Epochs",
                        )
                    }
                )

        full_predictions = np.array(full_predictions)

        return full_predictions
