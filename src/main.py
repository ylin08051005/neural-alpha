import os
from glob import glob
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig
from rich.progress import track
from torch.utils.data import DataLoader

from .dataset import preprocess, TrainAlphaDataset
from .trainer import Trainer
from .model.mlp import NeuralAlpha
from .model.loss import ICLoss
from .utils.constants import feature_names, label_name
from .utils.utils import seed_all, rolling

def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--symlink_path", type=str, default="data/symlink")
    parser.add_argument("--quick_expr", action="store_true", default=True)
    parser.add_argument("--train_scale", type=int, default=250)
    parser.add_argument("--look_back_window", type=int, default=20)
    parser.add_argument("--wandb_track", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)

    return parser.parse_args()


def init_wandb(config: DictConfig) -> None:
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config={
            "device": config.device,
            "optimizer": config.optimizer,
            "trainer": config.trainer,
            "data": config.data,
            "label_pretrainer": config.label_pretrainer,
        },
    )


def main(args: Namespace) -> None:
    seed_all(2025)
    folder_path = os.readlink(args.symlink_path)
    stock = {}

    for _, path in track(
        enumerate(glob(f"{folder_path}/*.csv")), description="Loading files..."
    ):
        curr_df = pd.read_csv(path)
        stock_name = os.path.basename(path).split(".")[0]

        if len(curr_df):
            stock[stock_name] = curr_df

    if args.quick_expr:
        subset = stock["2330"]
    else:
        subset = stock

    preprocessed = preprocess(subset, "sharpe", 5)

    first_train = torch.tensor(
        preprocessed.iloc[: args.train_scale][feature_names + label_name].to_numpy()
    )
    test = torch.tensor(
        preprocessed.iloc[args.train_scale :][feature_names + label_name].to_numpy()
    )

    window_data = rolling(first_train, args.look_back_window)
    test_window_data = rolling(test, args.look_back_window)
    first_train_dataset = TrainAlphaDataset(window_data)
    first_train_loader = DataLoader(first_train_dataset, batch_size=1, shuffle=True)

    model = NeuralAlpha(len(feature_names), 64, 1, 0.1)
    criterion = ICLoss(correlation_type="spearman")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = Trainer(model, criterion)
    trainer.train(args.n_epochs, first_train_loader, optimizer)
    trainer.trade_pipeline(
        args.threshold,
        first_train,
        torch.tensor(test_window_data[:, :, :-1], dtype=torch.float32),
        torch.tensor(test_window_data[:, :, -1], dtype=torch.float32),
        args.look_back_window,
        args.n_epochs,
        optimizer,
    )


if __name__ == "__main__":
    main(get_args())
