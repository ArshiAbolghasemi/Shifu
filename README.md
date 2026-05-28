# Shifu — DeepLOB for Chinese Equities

A research pipeline that applies DeepLOB (deep convolutional neural networks for limit order books) to Chinese A-share stocks. The model ingests 50 days of intraday order-flow and daily price history per asset, predicts tomorrow's return direction, and drives a rule-based long/short portfolio backtested on held-out data.

---

## Table of Contents

1. [Setup](#setup)
2. [Data](#data)
3. [End-to-End Pipeline](#end-to-end-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Labelling](#labelling)
6. [Model Architecture](#model-architecture)
7. [Training](#training)
8. [Portfolio Management](#portfolio-management)

---

## Setup

**Prerequisites:** Python ≥ 3.10, [uv](https://github.com/astral-sh/uv).

```bash
# 1. Install all dependencies (creates / reuses .venv automatically)
uv sync

# 2. Pull the data from the Cloudflare R2 remote
#    Credentials are stored in .dvc/config.local — contact the team for access
uv run dvc pull

# 3. Open JupyterLab and run notebooks in order
uv run jupyter lab
```

Linting is handled by **ruff** (`line-length = 100`). Run `uv run ruff check .` before committing.

---

## Data

Two parquet files are versioned with DVC (remote: Cloudflare R2 bucket `feishu`, ~2.2 GB total):

| File | Description |
|---|---|
| `data/lob_data_in_sample.parquet` | Tick-level LOB snapshots — 10 bid/ask price+volume levels per tick, timestamped |
| `data/daily_data_in_sample.parquet` | Daily OHLCV per asset, including `vwap_0930_0935` (the day-open execution reference price) |

The LOB file is streamed in 500 k-row batches rather than loaded into memory at once.

---

## End-to-End Pipeline

```mermaid
flowchart TD
    A["LOB Parquet\n(tick-level)"] --> B["Stream & Group\nby asset"]
    B --> C["Compute OFI\nper tick, 10 levels"]
    C --> D["Snap to 24 intraday slots\n09:40–15:00, 10-min grid"]
    D --> E["5-day causal\nrolling z-score\n→ 240 OFI features / day"]

    F["OHLCV Parquet\n(daily)"] --> G["Engineer 14 features\n(momentum, vol, liquidity…)"]
    G --> H["Cross-sectional\nz-score per day\n→ 19 OHLCV features / day"]

    E --> I["Merge on asset_id × date\n→ 259 features / asset-day"]
    H --> I
    I --> J["Build Labels\n(close_{t+1} − vwap_t) / vwap_t vs ±α"]
    J --> K["Slide T=50 day windows\nper asset"]
    K --> L["HDF5 Cache\ntrain / val / test splits"]

    L --> M["DeepLOB\nTraining"]
    M --> N["Checkpoint\nbest_model.pt"]
    N --> O["Signal Generation\nP(down / flat / up)"]
    O --> P["Portfolio Engine\nBuy / Hold / Sell"]
    P --> Q["Backtest Metrics\nCAGR · Sharpe · MDD"]
```

---

## Feature Engineering

### OFI Features (240)

Order-flow imbalance (OFI) measures the net pressure exerted by market participants at each price level of the order book. At each LOB level *i*, OFI is computed tick-by-tick as the change in bid-side depth minus the change in ask-side depth, where the sign convention accounts for whether the best price moved up, stayed, or moved down.

The pipeline:
1. Computes OFI for all 10 levels on every tick within a trading day.
2. Maps each tick to one of **24 fixed intraday time slots** (09:40, 09:50, …, 11:20, then 13:00–15:00 at 10-minute intervals). Off-grid ticks are discarded; missing slots are zero-filled.
3. Produces a `(24 slots × 10 levels) = 240`-dimensional daily vector, ordered slot-major.
4. Applies a **causal 5-day rolling z-score** per feature (using only past days, never the current or future), so no look-ahead leakage is introduced.

### OHLCV Features (19)

14 engineered features are computed per-asset, then all 19 (including the 5 raw prices) are **cross-sectionally z-scored per trading day** — meaning each feature is standardized across all assets on the same day, removing market-wide level effects.

| Group | Features |
|---|---|
| Momentum | `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d` |
| Realized volatility | `vol_5d`, `vol_20d` (rolling std of log-returns) |
| Liquidity | `amihud` (|return| / value traded), `volume_zscore` (20-day rolling) |
| Momentum oscillator | `rsi_14` |
| Price structure | `ma_dist_5`, `ma_dist_20` (distance from moving averages), `open_close_ret`, `high_low_range`, `close_vwap_dist` |
| Raw prices | `open`, `close`, `volume`, `low`, `high` |

The final input to the model is a **259-feature vector** per (asset, day): 240 OFI + 19 OHLCV, concatenated in that order. All values are clipped to [−5, 5] before being stored.

---

## Labelling

The prediction target is the forward return from entering a position at the day-*t* VWAP (09:30–09:35 average) and exiting at the day-*(t+1)* close:

```
return_t = (close_{t+1} − vwap_t) / |vwap_t|
```

This return is discretized into three classes using a symmetric threshold **α**:

```mermaid
flowchart LR
    R["return_t"]
    R -->|"< −α"| D["0 — Down"]
    R -->|"−α ≤ · ≤ +α"| F["1 — Flat"]
    R -->|"> +α"| U["2 — Up"]
```

Two checkpoints use different alpha values:

| Version | α | Description |
|---|---|---|
| v2 | 0.015 (1.5%) | Wider threshold, fewer signal days |
| v3 | 0.002 (0.2%) | Tighter threshold, used in production backtest |

The train/val/test split is **per-asset chronological**: 70% train → 85% cumulative val → last 15% test. No asset ever leaks future data into training.

---

## Model Architecture

DeepLOB treats each sample as a single-channel 2D image of shape `(1, T=50, NF=259)` — 50 trading days on the time axis and 259 features on the spatial axis. Three convolutional blocks progressively collapse the spatial dimension to 1, an Inception module captures multi-scale temporal patterns, and an LSTM reads the resulting sequence. An asset embedding is concatenated before the classification head to give the model per-stock context.

```mermaid
flowchart TD
    IN["Input\n1 × 50 × 259"]

    subgraph CB1["Conv Block 1 — LeakyReLU"]
        C1A["Conv2d 1→32, kernel (1,2), stride (1,2)\n→ 32 × 50 × 129"]
        C1B["Conv2d 32→32, kernel (4,1)\n→ 32 × 47 × 129"]
        C1C["Conv2d 32→32, kernel (4,1)\n→ 32 × 44 × 129"]
    end

    subgraph CB2["Conv Block 2 — Tanh"]
        C2A["Conv2d 32→32, kernel (1,2), stride (1,2)\n→ 32 × 44 × 64"]
        C2B["Conv2d 32→32, kernel (4,1)\n→ 32 × 41 × 64"]
        C2C["Conv2d 32→32, kernel (4,1)\n→ 32 × 38 × 64"]
    end

    subgraph CB3["Conv Block 3 — LeakyReLU"]
        C3A["Conv2d 32→32, kernel (1,64)\n→ 32 × 38 × 1"]
        C3B["Conv2d 32→32, kernel (4,1)\n→ 32 × 35 × 1"]
        C3C["Conv2d 32→32, kernel (4,1)\n→ 32 × 32 × 1"]
    end

    subgraph INC["Inception Module"]
        I1["Branch 1: 1×1 → 3×1 conv\n64 channels"]
        I2["Branch 2: 1×1 → 5×1 conv\n64 channels"]
        I3["Branch 3: MaxPool 3×1 → 1×1 conv\n64 channels"]
        CAT["Concat → 192 × 32 × 1"]
    end

    LSTM["LSTM  192 → 64\nlast hidden state"]
    EMB["Asset Embedding\nn_assets × embed_dim"]
    CAT2["Concat  64 + embed_dim"]
    FC["Linear → 3\nSoftmax"]
    OUT["P(down) · P(flat) · P(up)"]

    IN --> CB1 --> CB2 --> CB3 --> INC
    I1 & I2 & I3 --> CAT
    CAT --> LSTM
    EMB --> CAT2
    LSTM --> CAT2
    CAT2 --> FC --> OUT
```

### Key design choices

- **Spatial dimension = features, not price levels.** Unlike the original DeepLOB paper (which operates on raw bid/ask columns), this implementation treats the full 259-feature vector as the spatial axis, letting convolutions learn cross-feature interactions.
- **Stride-2 convolutions halve the spatial dimension.** Block 1 takes 259 → 129; block 2 takes 129 → 64; block 3 collapses the remaining 64 to 1 with a `(1, 64)` kernel. This avoids pooling and keeps gradient flow clean.
- **Inception branches capture short, medium, and pooled temporal context** at the 1-wide spatial stage (kernel heights 3, 5, and MaxPool-3).
- **Asset embedding** (with `max_norm=1.0`) is injected after the LSTM, giving the model a learnable, norm-bounded per-stock bias without contaminating the convolutional feature extraction.
- **BatchNorm after every conv** stabilizes training across the varied scale of OFI and OHLCV features.

---

## Training

| Hyperparameter | Value |
|---|---|
| Lookback window T | 50 trading days |
| Batch size | 64 |
| Optimizer | Adam, lr = 1e-4 |
| Loss | CrossEntropyLoss |
| Epochs | 10 |
| Checkpoint frequency | every 5 epochs + best val-loss |

### Data pipeline for training

Because the full sliding-window dataset (millions of `50 × 259` float32 arrays) does not fit in RAM, it is written once to an **HDF5 cache** (`cache/daily_windows/{train,val,test}.h5`) using an append-mode writer. At training time `DiskLOBDataset` opens the HDF5 file lazily per DataLoader worker, reading individual samples on demand. This keeps the peak RAM footprint proportional to a single batch, not the entire dataset.

```mermaid
sequenceDiagram
    participant NB as model.ipynb
    participant W  as WindowWriter
    participant H5 as HDF5 Cache
    participant DS as DiskLOBDataset
    participant M  as DeepLOB

    NB->>W: open train/val/test writers
    loop per asset
        NB->>W: write(X_windows, labels, asset_idx)
        W->>H5: append rows to datasets
    end
    NB->>DS: DiskLOBDataset("train")
    DS->>H5: read length
    loop each epoch
        M->>DS: __getitem__(idx)
        DS->>H5: lazy read X[idx], y[idx]
        DS-->>M: tensor (1, T, NF)
    end
```

The best checkpoint (lowest validation loss) is saved as `best_model.pt`; full optimizer state is saved every 5 epochs for resumability.

---

## Portfolio Management

`notebooks/portfolio_management.ipynb` backtests the trained model strictly on the **test split** (the last 15% of trading days seen by no training epoch).

### Signal generation

For each test day *t*, the model receives the `(n_valid_assets, 1, T, NF)` feature tensor covering the window `[t−T+1, t]` and outputs a probability triplet `[P(down), P(flat), P(up)]` for every asset. Assets without a complete T-day history are excluded on that day.

### Decision rules

```mermaid
flowchart TD
    SIG["Model output per asset:\nP_down · P_flat · P_up"]

    SIG --> SELL_CHECK{"Held position?\nP_down > 0.50\nand signal = −1"}
    SELL_CHECK -->|Yes, up to top-10| FULL_SELL["Full sell\nat close price\n(or open if sell_mode=open)"]
    SELL_CHECK -->|signal = 0 flat| PART_SELL["Partial sell\n50% of position"]

    SIG --> BUY_CHECK{"P_up > 0.50\nand signal = +1\nnot sold today"}
    BUY_CHECK -->|Top-20 assets| BUY["Buy at VWAP\n(09:30–09:35)\nequal weight"]

    FULL_SELL & PART_SELL & BUY --> PORT["Update portfolio\nT+1 rule enforced"]
```

**Order of operations each day:**
1. **Sell first.** Full sells (strong down signal, top-10 by P(down)) are executed before buys, freeing capital. Partial sells (flat signal, 50% of position) follow.
2. **Buy second.** Top-20 assets by P(up) that weren't sold today and have P(up) > 0.50 receive equal allocations from the available cash, holding back a 5% cash buffer.
3. **T+1 lock.** Shares bought on day *t* are locked and cannot be sold until day *t+1*, simulating the Chinese A-share settlement rule.

### Execution and costs

| Parameter | Value |
|---|---|
| Initial capital | RMB 50,000,000 |
| Lot size | 100 shares |
| Commission (buy & sell) | 0.01% (min RMB 5) |
| Stamp duty (sell only) | 0.05% |
| Minimum holdings constraint | 10 positions at all times |
| Entry price | VWAP (09:30–09:35) |
| Exit price | Close (or Open if `sell_mode=open`) |

### Performance metrics

```mermaid
flowchart LR
    BT["Daily portfolio values"] --> CAGR["CAGR"]
    BT --> SR["Annualized Sharpe\n(rf = 1.72% p.a.)"]
    BT --> MDD["Maximum Drawdown"]
    CAGR & SR & MDD --> SCORE["Score proxy\n0.45×CAGR + 0.30×Sharpe + 0.25×(−MDD)"]
    BT --> AUX["Calmar · Volatility\nWin rate · Avg holdings\nTransaction costs"]
```

The notebook also writes a trade log to `submissions/` in CSV format (`trade_day_id`, `asset_id`, `buy_percentage`, `sell_percentage`) and validates it against the competition submission rules before saving.

---
