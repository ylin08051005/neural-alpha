import os
from glob import glob
from argparse import ArgumentParser, Namespace

import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

from .dataset import preprocess
from .trainer import MultiAlphaTrainer
from .model.nn_model import AlphaSelfAttention
from .model.loss import ICLoss, InverseICLoss
from .utils.utils import seed_all
from .pipeline import Pipeline


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--symlink_path", type=str, default="data/symlink")
    parser.add_argument("--folder_path", type=str)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--quick_expr", action="store_true", default=True)
    parser.add_argument("--train_scale", type=int, default=1500)
    parser.add_argument("--test_scale", type=int, default=125)
    parser.add_argument("--prediction_window", type=int, default=5)
    parser.add_argument("--direction", type=str, default="pos")
    parser.add_argument("--label_type", type=str, default="cumret")
    parser.add_argument("--wandb_track", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--loss_multiplier", type=float, default=1.0)

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


if __name__ == "__main__":
    args = get_args()
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
        subset, args.label_type, args.prediction_window, args.direction
    )
    pipeline = Pipeline(
        preprocessed,
        shortest_sequence,
        args.train_scale,
        args.test_scale,
        args.prediction_window,
    )
    model = AlphaSelfAttention(
        input_dim=pipeline.feat_size,
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

    model_path = (
        f"{args.prediction_window}d_cumret_10_alphas_eta_{args.loss_multiplier}.pt"
    )
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

    predictions = pipeline.run(
        model, trainer, model_path, stock_amount, args.loss_multiplier
    )
