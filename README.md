# Finlang - Crypto Trading Signal Generator

Lightweight trading signal generator using remote prediction API. Supports multiple strategies with automatic cold start via historical backfilling.

## Features

- **Multiple Strategies**: voting_48h (recommended), voting, voting_120h, mlp, dynamic
- **Auto Backfill**: Automatically fills missing history on cold start
- **Cron Ready**: Designed for hourly crontab execution
- **Webhook Support**: Send signals to external services
- **Pure Python**: Minimal dependencies (requests, numpy, pandas)

## Strategies

| Strategy | Window | Long Threshold | Short Threshold | Allows Short | Sharpe (Backtest) | vs B&H |
|----------|--------|----------------|-----------------|--------------|-------------------|--------|
| `voting_48h` | 48h | 58% | 42% | No | **1.45** | ✅ BEATS |
| `voting` | 96h | 50% | 40% | No | 0.65 | ❌ |
| `voting_120h` | 120h | 58% | 40% | No | 0.97 | ❌ |
| `mlp` | 272h* | 55% | 45% | No | TBD | TBD |
| `dynamic` | 48h | Adaptive | Adaptive | No | TBD | TBD |

\* MLP requires 200 training samples + 72h lookahead = 272 hours minimum

**Backtest period**: 2023-02-14 ~ 2026-01-29 (25,928 hours, 72h rebalance frequency)

**Buy & Hold baseline**: Sharpe 1.18, Return 262%

### Strategy Details

#### Voting 48h (Recommended - Beats B&H)
48h rolling window with 58% threshold. Only strategy that beats Buy & Hold in backtests.

```
If bullish_pct >= 58% → LONG
If bullish_pct < 42%  → FLAT
Otherwise             → FLAT
```

**Performance**: Sharpe 1.45, Return 275%, Max Drawdown -30%

#### Voting (96h)
Counts bullish predictions (upside_probability > 50%) in a 96h rolling window. Long-only.

```
If bullish_pct >= 50% → LONG
If bullish_pct < 40%  → FLAT
Otherwise             → FLAT
```

#### Voting 120h
Longer window with stricter threshold.

```
If bullish_pct >= 58% → LONG
If bullish_pct < 40%  → FLAT
Otherwise             → FLAT
```

#### MLP (Machine Learning)
Online learning neural network trained on historical upside_probability → 72h price change.

#### Dynamic
Adaptive thresholds based on recent volatility:
- High volatility: stricter thresholds (65%/35%)
- Low volatility: looser thresholds (52%/48%)

## Installation

```bash
# Clone repository
git clone https://github.com/quant-maker/finlang.git
cd finlang

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set environment variables:

```bash
# Required: Prediction API URL
export FINLANG_API_URL="https://your-prediction-service.example.com"

# Optional: Webhook for signals
export FINLANG_WEBHOOK_URL="https://your-webhook.example.com/signal"

# Optional: Custom signal output directory
export FINLANG_SIGNAL_DIR="/path/to/signals"
```

## Usage

### Manual Run

```bash
# Recommended strategy (voting_48h - beats B&H)
python -m finlang.cron --symbol BTCUSDT --strategy voting_48h

# Other strategies
python -m finlang.cron --symbol BTCUSDT --strategy voting
python -m finlang.cron --symbol BTCUSDT --strategy voting_120h
python -m finlang.cron --symbol BTCUSDT --strategy mlp
python -m finlang.cron --symbol BTCUSDT --strategy dynamic

# Test run (no file saving)
python -m finlang.cron --symbol BTCUSDT --strategy voting_48h --no-save --no-backfill

# JSON output
python -m finlang.cron --symbol BTCUSDT --json
```

### Backfill History (Cold Start)

First-time setup or recovering from data loss:

```bash
# Backfill 300 hours of history
python -m finlang.cron --symbol BTCUSDT --backfill 300

# For MLP strategy, backfill at least 272 hours
python -m finlang.cron --symbol BTCUSDT --backfill 300
```

Backfill time estimate: ~2 seconds per hour → 300 hours ≈ 10 minutes

## Crontab Deployment

### Setup Script

Create `/opt/finlang/run.sh`:

```bash
#!/bin/bash
set -e

# Configuration
export FINLANG_API_URL="https://your-prediction-service.example.com"
export FINLANG_WEBHOOK_URL="https://your-webhook.example.com/signal"  # Optional

# Activate virtual environment (if using)
source /opt/finlang/venv/bin/activate

# Run strategy
cd /opt/finlang
python -m finlang.cron --symbol BTCUSDT --strategy "$1"
```

```bash
chmod +x /opt/finlang/run.sh
```

### Crontab Configuration

Edit crontab: `crontab -e`

#### Single Strategy

```cron
# Run voting strategy every hour at minute 5
5 * * * * /opt/finlang/run.sh voting >> /var/log/finlang/voting.log 2>&1
```

#### Multiple Strategies

```cron
# Voting (96h, long-only) - every hour at minute 5
5 * * * * /opt/finlang/run.sh voting >> /var/log/finlang/voting.log 2>&1

# Voting 120h (best Sharpe) - every hour at minute 6
6 * * * * /opt/finlang/run.sh voting_120h >> /var/log/finlang/voting_120h.log 2>&1

# MLP (machine learning) - every hour at minute 7
7 * * * * /opt/finlang/run.sh mlp >> /var/log/finlang/mlp.log 2>&1

# Dynamic (adaptive thresholds) - every hour at minute 8
8 * * * * /opt/finlang/run.sh dynamic >> /var/log/finlang/dynamic.log 2>&1
```

#### With Log Rotation

```cron
# Run with logrotate-friendly output
5 * * * * /opt/finlang/run.sh voting 2>&1 | /usr/bin/logger -t finlang-voting
```

### Complete Deployment Example

```bash
# 1. Create directory structure
sudo mkdir -p /opt/finlang /var/log/finlang
sudo chown $USER:$USER /opt/finlang /var/log/finlang

# 2. Clone and setup
cd /opt/finlang
git clone https://github.com/quant-maker/finlang.git .
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Create run script
cat > /opt/finlang/run.sh << 'EOF'
#!/bin/bash
set -e
export FINLANG_API_URL="https://your-prediction-service.example.com"
source /opt/finlang/venv/bin/activate
cd /opt/finlang
python -m finlang.cron --symbol BTCUSDT --strategy "$1"
EOF
chmod +x /opt/finlang/run.sh

# 4. Initial backfill (cold start)
source venv/bin/activate
export FINLANG_API_URL="https://your-prediction-service.example.com"
python -m finlang.cron --symbol BTCUSDT --backfill 300

# 5. Test run
./run.sh voting_120h

# 6. Setup crontab
(crontab -l 2>/dev/null; echo "5 * * * * /opt/finlang/run.sh voting_120h >> /var/log/finlang/voting_120h.log 2>&1") | crontab -
```

### Verify Deployment

```bash
# Check crontab
crontab -l

# Monitor logs
tail -f /var/log/finlang/voting_120h.log

# Check latest signal
cat /opt/finlang/signals/btcusdt_voting_120h_latest.json
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
  "strategy": "voting_120h",
  "source": "prediction_api",
  "factors": {
    "sig_24h": 65.0,
    "history_count": 120,
    "bullish_pct": 58.3,
    "bullish_count": 70,
    "total_count": 120,
    "allow_short": false
  }
}
```

### Position Values

| Position | Action | Meaning |
|----------|--------|---------|
| `1` | LONG | Buy / Hold long |
| `0` | FLAT | No position / Exit |

## File Structure

```
finlang/
├── __init__.py          # Package init
├── api.py               # Prediction API client
├── cron.py              # Main cron script
├── base.py              # Signal base class
├── model/
│   ├── __init__.py
│   └── mlp.py           # Pure NumPy MLP
├── requirements.txt     # Dependencies
├── history/             # Auto-created: prediction history
│   └── btcusdt/
│       └── 20260131_1400.json
├── signals/             # Auto-created: output signals
│   └── btcusdt_voting_120h_latest.json
└── models/              # Auto-created: MLP weights
    └── mlp_btcusdt.npz
```

## Troubleshooting

### "FINLANG_API_URL not set"

Set the environment variable:
```bash
export FINLANG_API_URL="https://your-prediction-service.example.com"
```

### "History insufficient"

Run backfill:
```bash
python -m finlang.cron --symbol BTCUSDT --backfill 300
```

### "API timeout"

Check prediction service availability. Use `--no-backfill` to skip auto-backfill:
```bash
python -m finlang.cron --symbol BTCUSDT --no-backfill
```

### Cron not running

1. Check crontab: `crontab -l`
2. Check cron logs: `grep CRON /var/log/syslog`
3. Test script manually: `/opt/finlang/run.sh voting`
4. Ensure environment variables are set in script

## License

MIT
