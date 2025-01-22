# Neural Alpha

Make non-sense formulaic alphas more non-sense

## Installation

### Install via `uv` (Recommends)

Make sure to have uv installed on your system

```bash
uv init && uv venv
source .venv/bin/activate
uv sync pyproject.toml
```

### Install via `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Download data

```bash
chmod +x scripts/prepare_data.sh
bash scripts/prepare_data.sh
```

If successful, you should see `data/tw_stock_data` and there should be tons of `.csv` files inside.

### Execute training script

If using symbolic link

```bash
python -m src.main \
    --symlink_path data/symlink \
    --quick_expr \
    --train_scale 500 \
    --look_back_window 60 \
    --direction pos \
    --label_type cumret \
    --future_window 20 \
    --n_epochs 100 \
    --batch_size 1
```

If not

```bash
python -m src.main \
    --folder_path data/tw_stock_data \
    --quick_expr \
    --train_scale 500 \
    --look_back_window 60 \
    --direction pos \
    --label_type cumret \
    --future_window 20 \
    --n_epochs 100 \
    --batch_size 1
```

### Run testing with `notebooks/see_rankvec.ipynb`
