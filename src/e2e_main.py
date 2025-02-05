import os
from argparse import ArgumentParser, Namespace
from glob import glob

import pandas as pd
import torch
from omegaconf import OmegaConf
from rich.progress import track

from .dataset import preprocess
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


if __name__ == "__main__":
    args = get_args()
    print(args.__dict__)
    stock_conf = OmegaConf.load("config/selected_stocks.yaml")
    seed_all(2025)
    wandb_conf = OmegaConf.load("config/wandb.yaml")

    if args.wandb_track:
        wandb_conf.batch_size = args.batch_size
        wandb_conf.epochs = args.n_epochs

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
        device,
    )

    predictions = pipeline.run(
        args.wnadb_track,
        args.n_epochs,
        stock_amount,
        args.batch_size,
        args.loss_multiplier,
        wandb_config=wandb_conf,
    )
