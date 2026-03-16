# Optiver Realized Volatility Prediction

Kaggle Code Competition pipeline: predict short-term realized volatility from order book and trade data.

## Project structure

| File | Role |
|------|------|
| `utils.py` | RMSPE metric, data paths (Kaggle vs local), `make_row_id` |
| `data_loading.py` | Load `train.csv`, `test.csv`, `book_*.parquet`, `trade_*.parquet` |
| `feature_engineering.py` | Build full-window + sub-window book/trade features, including realized-volatility features |
| `train.py` | Grouped CV by `time_id`, multi-seed LightGBM ensemble, save models + OOF + reports |
| `inference.py` | Build test features, average saved models, write submission file |

## Data (Kaggle)

- **train.csv**: `stock_id`, `time_id`, `target`
- **test.csv**: `stock_id`, `time_id`, `row_id`
- **book_train.parquet / book_test.parquet**: L1/L2 book (bid/ask price & size)
- **trade_train.parquet / trade_test.parquet**: price, size, order_count

Feature window = first 10 minutes of each 20-minute bucket; target = realized vol over next 10 minutes.

## Run locally

Use this project Python everywhere:

```bash
/Users/doraweng/miniconda3/bin/python
```

**Option A – with real Kaggle data:**  
Download competition data and put `train.csv`, `test.csv`, `book_*.parquet`, `trade_*.parquet` in the project directory (or in `synthetic_data/`). Then:

```bash
/Users/doraweng/miniconda3/bin/python -m pip install -r requirements.txt
/Users/doraweng/miniconda3/bin/python train.py
/Users/doraweng/miniconda3/bin/python inference.py
```

**Option B – without parquet (synthetic subset):**  
Generates minimal book/trade parquet from train/test CSV so the pipeline runs end-to-end:

```bash
/Users/doraweng/miniconda3/bin/python -m pip install -r requirements.txt
/Users/doraweng/miniconda3/bin/python create_synthetic_data.py
/Users/doraweng/miniconda3/bin/python train.py
/Users/doraweng/miniconda3/bin/python inference.py
```

Or one command:

```bash
/Users/doraweng/miniconda3/bin/python run_all.py
```

This writes outputs to:

- `artifacts/models/`
- `artifacts/oof/`
- `artifacts/reports/`
- `artifacts/submissions/`

**Note (macOS):** If `lightgbm` fails with `libomp.dylib`, install OpenMP first:

```bash
brew install libomp
```

On Kaggle Notebooks this is not needed.

## Create A Local Holdout Test Set

If you want a real local test split from training data, create a grouped holdout
set by `time_id`:

```bash
/Users/doraweng/miniconda3/bin/python create_holdout_split.py
```

This creates a `local_holdout/` directory with:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `holdout_targets.csv`
- `book_train.parquet`
- `book_test.parquet`
- `trade_train.parquet`
- `trade_test.parquet`

You can also control the split size:

```bash
/Users/doraweng/miniconda3/bin/python create_holdout_split.py --holdout-fraction 0.2 --seed 42
```

`holdout_targets.csv` contains the true targets for the local test set, so you can
compare your predictions locally after running inference on `local_holdout/`.

To use the generated split from Python:

```python
from train import run_train
from inference import run_inference

run_train(data_dir="local_holdout", output_dir="local_holdout")
run_inference(data_dir="local_holdout", model_dir="local_holdout/artifacts")
```

## Run on Kaggle Notebook

1. Add competition data: **Optiver Realized Volatility Prediction**.
2. Copy in: `utils.py`, `data_loading.py`, `feature_engineering.py`, `train.py`, `inference.py`.
3. In a cell:
   ```python
   from train import run_train
   from inference import run_inference
   run_train(output_dir=".")
   run_inference(model_dir="artifacts", output_path="submission.csv")
   ```
4. Set output to **submission.csv** and submit.

## Submission format

- File name: **submission.csv**
- Header: **row_id,target**
- Rows: e.g. `0-4,0.003` (one per test row, same order as test.csv).

## Evaluation

- **RMSPE** (Root Mean Square Percentage Error). Validation in `train.py` uses grouped splits by `time_id`, reports fold RMSPE, and saves OOF/report artifacts.

## Minimum viable baseline

- Load book/trade, aggregate by `(stock_id, time_id)`.
- Features include WAP/spread/imbalance plus log-return realized-volatility and sub-window features.
- Multi-seed LightGBM ensemble with grouped CV by `time_id`.
- Output exactly `row_id,target` in `submission.csv`.

## Important feature ideas

- **Volatility proxies**: WAP return std in window, high-low, trade price std.
- **Liquidity**: spread, total size, depth (L2).
- **Imbalance**: size and price imbalance L1/L2.
- **Trade activity**: volume, order count, trade intensity over time.
- **Stock/time id**: stock_id or time_id as categorical (if many stocks).

## Common mistakes

- **Wrong metric**: Optimizing RMSE/MAE instead of RMSPE (validation should use RMSPE).
- **Leakage**: Using the same `time_id` in train and val; use grouped/block splits by `time_id`.
- **Submission format**: Extra columns, wrong column order, or wrong filename; must be exactly `submission.csv` with `row_id,target`.
- **Row order**: Predictions must align 1:1 with test.csv (same row_id order); we use test’s `row_id` column.
- **Missing parquet**: On Kaggle, ensure book/trade parquet files are in the input dataset; locally, ensure paths in `utils.get_data_dir()` point to them.
