# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shifu is a financial AI research project implementing **DeepLOB** — a deep convolutional model for limit order book (LOB) data — to predict intraday returns on Chinese stocks. The model outputs P(down/flat/up) per asset and feeds a backtested portfolio strategy.

## Environment & Commands

This project uses `uv` for dependency management and a `.venv` virtual environment.

```bash
# Install dependencies
uv sync

# Run JupyterLab
uv run jupyter lab

# Lint notebooks and Python files
uv run ruff check .
uv run ruff format .

# Pull data from Cloudflare R2 remote (credentials in .dvc/config.local)
uv run dvc pull
```

Linting: `ruff` is configured with `line-length = 100`; notebooks ignore `E402` (import order).

## Architecture

### Data

Two parquet files are versioned via DVC (remote: Cloudflare R2 bucket `feishu`):
- `data/lob_data_in_sample.parquet` — intraday LOB snapshots (10 bid/ask price+volume levels, 24 slots/day at 10-min grid 09:40–15:00)
- `data/daily_data_in_sample.parquet` — daily OHLCV + engineered features per asset

### Feature Engineering (259 features per asset-day)

- **240 OFI features**: order-flow imbalance at every intraday slot × LOB level (24 slots × 10 levels), 5-day causal rolling z-score per feature
- **19 OHLCV features**: 14 engineered (momentum, volatility, liquidity, price-structure) + 5 raw (open, close, volume, low, high), cross-sectionally z-scored per day

### Labels

`label = sign((close_{t+1} − vwap_t) / vwap_t)` discretized at ±`ALPHA` → 3 classes (0=down, 1=flat, 2=up). `ALPHA=0.002` is used in production checkpoints (`v3`); `ALPHA=0.015` in earlier v2 experiments.

### Model: DeepLOB

Input shape: `(1, T=50, NF=259)` — a 50-day lookback window of per-day feature vectors.  
To avoid loading all windows into RAM, the data pipeline builds an **HDF5 cache** (`cache/daily_windows/{train,val,test}.h5`) once via `WindowWriter`, then serves it via `DiskLOBDataset` (lazy, multiprocessing-safe).

### Notebooks (primary entry points)

| Notebook | Purpose |
|---|---|
| `notebooks/lob_eda.ipynb` | EDA on LOB and daily parquet schemas, intraday patterns, OFI/spread analysis |
| `notebooks/alpha_signal.ipynb` | Sweep ALPHA thresholds (0.1%–1.5%) to pick the label cutoff |
| `notebooks/model.ipynb` | Full training pipeline: data ingestion → HDF5 cache → DeepLOB training → checkpoint |
| `notebooks/portfolio_management.ipynb` | Backtest on the test split; loads a checkpoint and runs a simulated portfolio |

### Checkpoints

```
checkpoints/deeplob/v2/best_model_alpha_0015.pt   ← ALPHA=0.015
checkpoints/deeplob/v3/best_model_alpha_0002.pt   ← ALPHA=0.002 (production)
checkpoints/deeplob/v3/best_model_alpha_0015.pt
```

### Portfolio Backtest Rules

- Initial capital: 50M CNY; lot size: 100 shares
- Commission: 1e-4; stamp duty: 5e-4; min commission: 5 CNY
- Buy at `vwap_0930_0935`, sell at `close` (T+1 rule enforced)
- Train/val/test chronological split: 70% / 15% / 15% per asset

## Device Handling

The notebooks auto-detect device: CUDA → MPS (Apple Silicon) → CPU. `NUM_WORKERS=0` and `PIN_MEMORY=False` are enforced on MPS/CPU to avoid dataloader issues.
