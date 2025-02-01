import os
from glob import glob
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from torch.utils.data import DataLoader

from .dataset import preprocess, TrainAlphaDataset
from .trainer import Trainer, MultiAlphaTrainer
from .model.nn_model import NeuralAlpha, AlphaSelfAttention
from .model.loss import ICLoss, InverseICLoss
from .utils.utils import dict2mat, seed_all, rolling


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--symlink_path", type=str, default="data/symlink")
    parser.add_argument("--folder_path", type=str)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--quick_expr", action="store_true", default=True)
    parser.add_argument("--train_scale", type=int, default=561)
    parser.add_argument("--look_back_window", type=int, default=60)
    parser.add_argument("--direction", type=str, default="pos")
    parser.add_argument("--label_type", type=str, default="cumret")
    parser.add_argument("--future_window", type=int, default=5)
    parser.add_argument("--wandb_track", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--loss_multiplier", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.1)

    return parser.parse_args()


def init_wandb(config: DictConfig) -> None:
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config={
            "learning_rate": config.lr,
            "architecture": "self_attention",
            "batch_size": config.batch_size,
            "epochs": config.epochs,
        },
    )


def main(args: Namespace) -> None:
    print(args.__dict__)
    stock_conf = OmegaConf.load("config/selected_stocks.yaml")
    seed_all(2025)

    if args.wandb_track:
        wandb_conf = OmegaConf.load("config/wandb.yaml")
        wandb_conf.batch_size = args.batch_size
        wandb_conf.epochs = args.n_epochs
        init_wandb(wandb_conf)

    if args.symlink_path:
        folder_path = os.readlink(args.symlink_path)
    else:
        folder_path = args.folder_path

    device = f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu"
    stock = {}

    for _, path in track(
        enumerate(glob(f"{folder_path}/*.csv")), description="Loading files..."
    ):
        curr_df = pd.read_csv(path)
        stock_name = os.path.basename(path).split(".")[0]

        if len(curr_df) >= 4000:
            stock[stock_name] = curr_df

    if args.quick_expr:
        subset = {k: stock[k] for k in list(stock_conf.for_expr)}
    else:
        subset = stock

    stock_amount = len(subset)
    shortest_sequence = min([len(v) for v in subset.values()])
    preprocessed = preprocess(
        subset, args.label_type, args.future_window, args.direction
    )
    first_train, test = dict2mat(preprocessed, args.train_scale, shortest_sequence)
    first_train = np.transpose(first_train, (1, 0, 2))  # (days, stocks, features)
    test = np.transpose(test, (1, 0, 2))  # (days, stocks, label)

    tr_window_feat, tr_window_label = rolling(
        first_train, args.look_back_window
    )  # (train_scale - lbw - 1, n_stocks, n_feat * lbw), (train_scale - lbw - 1, n_stocks, 1)
    ts_window_feat, ts_window_label = rolling(test, args.look_back_window)
    first_train_dataset = TrainAlphaDataset(tr_window_feat, tr_window_label)
    first_train_loader = DataLoader(
        first_train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # model = NeuralAlpha(tr_window_feat.shape[-1], 128, 1, 0.1).to(device)
    model = AlphaSelfAttention(
        input_dim=tr_window_feat.shape[-1],
        embed_dim=128,
        num_heads=1,
        dropout=0.1,
        kdim=None,
        vdim=None,
        final_output_dim=10,
        device=device,
        dtype=torch.float32,
        value_weight_type="vanilla",
    ).to(device)

    if args.wandb_track:
        wandb.watch(model, log="all", log_freq=10)

    label_ic = ICLoss(ic_type="spearman")
    pool_ic = InverseICLoss(ic_type="spearman")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model_path = f"vanilla_attn_epoch_{args.n_epochs}_bs{args.batch_size}_tr_{args.train_scale}_lbw_{args.look_back_window}_fw_{args.future_window}{args.direction}{args.label_type}.pt"
    # trainer = Trainer(model, "attention", label_ic, device, model_path)
    trainer = MultiAlphaTrainer(
        model,
        model_type="attention",
        label_loss=label_ic,
        pool_loss=pool_ic,
        device=device,
        model_path=model_path,
    )

    if args.wandb_track:
        trainer.wandb = wandb

    trainer.multi_asset_train(
        args.n_epochs,
        first_train_loader,
        optimizer,
        loss_multiplier=args.loss_multiplier,
        ndcg_k=stock_amount,
    )

    test = False
    if test:
        trainer.trade_pipeline(
            args.threshold,
            tr_window_feat,
            tr_window_label,
            ts_window_feat,
            ts_window_label,
            args.look_back_window,
            args.n_epochs,
            optimizer,
        )


if __name__ == "__main__":
    main(get_args())
