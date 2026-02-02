# Finlang - Crypto Trading Signal Generator

Lightweight trading signal generator using remote prediction API. Supports parameterized voting strategy with automatic cold start via historical backfilling.

## Features

- **Parameterized Voting Strategy**: Customize window and threshold via CLI
- **Auto Backfill**: Automatically fills missing history on cold start
- **Cron Ready**: Designed for hourly crontab execution
- **Webhook Support**: Send signals to external services
- **Pure Python**: Minimal dependencies (requests, numpy, pandas)

## Strategies

| Strategy | Description | Default Params |
|----------|-------------|----------------|
| `voting` | Count bullish votes in rolling window | window=48, threshold=0.58 |
| `mlp` | Online learning neural network | - |
| `dynamic` | Adaptive thresholds based on volatility | - |

### Voting Strategy Parameters

| Param | CLI Flag | Default | Description |
|-------|----------|---------|-------------|
| window | `--window` | 48 | Rolling window in hours |
| threshold | `--threshold` | 0.58 | Bullish % to go LONG (0-1) |

### Backtest Results (2023-02-14 ~ 2026-01-29, 72h rebalance)

| Window | Threshold | Sharpe | Return | vs B&H |
|--------|-----------|--------|--------|--------|
| **48h** | **0.58** | **1.45** | **275%** | ✅ **BEATS** |
| 48h | 0.60 | 1.56 | 277% | ✅ BEATS |
| 96h | 0.50 | 0.65 | 116% | ❌ |
| 120h | 0.58 | 0.97 | 185% | ❌ |

**Buy & Hold baseline**: Sharpe 1.18, Return 262%

## Installation

```bash
git clone https://github.com/quant-maker/finlang.git
cd finlang
pip install -r requirements.txt
```

## Configuration

Set environment variables:

```bash
# Required: Prediction API URL
export FINLANG_API_URL="https://your-prediction-service.example.com"

# Optional: Webhook for signals
export FINLANG_WEBHOOK_URL="https://your-webhook.example.com/signal"
```

## Usage

### Recommended (Beats B&H)

```bash
# Default: 48h window, 58% threshold (Sharpe 1.45)
python -m finlang.cron --symbol BTCUSDT

# Explicit params
python -m finlang.cron --symbol BTCUSDT --strategy voting --window 48 --threshold 0.58
```

### Custom Parameters

```bash
# 96h window, 50% threshold
python -m finlang.cron --symbol BTCUSDT --window 96 --threshold 0.50

# 120h window, 58% threshold
python -m finlang.cron --symbol BTCUSDT --window 120 --threshold 0.58
```

### Other Strategies

```bash
python -m finlang.cron --symbol BTCUSDT --strategy mlp
python -m finlang.cron --symbol BTCUSDT --strategy dynamic
```

### Test Run

```bash
# No file saving, no backfill
python -m finlang.cron --symbol BTCUSDT --no-save --no-backfill

# JSON output
python -m finlang.cron --symbol BTCUSDT --json
```

### Backfill History (Cold Start)

```bash
# Backfill 100 hours of history
python -m finlang.cron --symbol BTCUSDT --backfill 100

# For MLP strategy, backfill at least 272 hours
python -m finlang.cron --symbol BTCUSDT --backfill 300
```

## Crontab Deployment

### Setup Script

Create `/opt/finlang/run.sh`:

```bash
#!/bin/bash
set -e
export FINLANG_API_URL="https://your-prediction-service.example.com"
source /opt/finlang/venv/bin/activate
cd /opt/finlang
python -m finlang.cron --symbol BTCUSDT --window 48 --threshold 0.58
```

### Crontab

```cron
# Run every hour at minute 5
5 * * * * /opt/finlang/run.sh >> /var/log/finlang/cron.log 2>&1
```

## Output

### Signal Format

```json
{
  "timestamp": "2026-01-31T14:00:00+00:00",
  "symbol": "BTCUSDT",
  "position": 1,
  "action": "LONG",
  "trend": "bullish",
  "score": 0.583,
  "current_close": 82500.0,
  "strategy": "voting",
  "factors": {
    "sig_24h": 65.0,
    "history_count": 48,
    "bullish_pct": 58.3,
    "bullish_count": 28,
    "total_count": 48,
    "window": 48,
    "threshold": 0.58
  }
}
```

### Position Values

| Position | Action | Meaning |
|----------|--------|---------|
| `1` | LONG | Buy / Hold long |
| `0` | FLAT | No position / Exit |

## CLI Reference

```
python -m finlang.cron --help

Options:
  --symbol, -s      Trading symbol (default: BTCUSDT)
  --strategy, -t    Strategy type: voting, mlp, dynamic (default: voting)
  --window, -w      Voting window in hours (default: 48)
  --threshold, -th  Voting threshold 0-1 (default: 0.58)
  --api-url         Prediction API URL
  --signal-dir      Signal output directory
  --webhook         Webhook URL for signals
  --backfill N      Backfill N hours of history
  --force-retrain   Force MLP retraining
  --no-save         Don't save signal files
  --no-backfill     Disable auto-backfill
  --json            Output as JSON
```

## License

MIT
