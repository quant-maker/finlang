# Finlang - Crypto Trading Signal Generator

Lightweight trading signal generator using remote prediction API. Supports multiple strategies with automatic cold start via historical backfilling.

## Features

- **Multiple Strategies**: voting, voting_72h, mlp, dynamic
- **Auto Backfill**: Automatically fills missing history on cold start
- **Cron Ready**: Designed for hourly crontab execution
- **Webhook Support**: Send signals to external services
- **Pure Python**: Minimal dependencies (requests, numpy, pandas)

## Strategies

| Strategy | Window | Long Threshold | Short Threshold | Allows Short | Expected Sharpe |
|----------|--------|----------------|-----------------|--------------|-----------------|
| `voting` | 96h | 55% | 40% | No | ~0.95 |
| `voting_72h` | 72h | 52% | 48% | **Yes** | ~1.02 |
| `mlp` | 272h* | 55% | 45% | No | ~0.90 |
| `dynamic` | 48h | Adaptive | Adaptive | No | ~0.85 |

\* MLP requires 200 training samples + 72h lookahead = 272 hours minimum

### Strategy Details

#### Voting (Default)
Counts bullish predictions (sig_24h > 50%) in a rolling window. Conservative, long-only.

```
If bullish_pct >= 55% → LONG
If bullish_pct < 40%  → FLAT
Otherwise             → FLAT
```

#### Voting 72h (Best Sharpe)
Shorter window with tighter thresholds. Supports short positions.

```
If bullish_pct >= 52% → LONG
If bullish_pct < 48%  → SHORT
Otherwise             → FLAT
```

#### MLP (Machine Learning)
Online learning neural network trained on historical sig_24h → 72h price change.

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
# Default strategy (voting)
python -m finlang.cron --symbol BTCUSDT

# Specific strategy
python -m finlang.cron --symbol BTCUSDT --strategy voting_72h
python -m finlang.cron --symbol BTCUSDT --strategy mlp
python -m finlang.cron --symbol BTCUSDT --strategy dynamic

# Test run (no file saving)
python -m finlang.cron --symbol BTCUSDT --strategy voting_72h --no-save --no-backfill

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

# Voting 72h (best Sharpe, supports short) - every hour at minute 6
6 * * * * /opt/finlang/run.sh voting_72h >> /var/log/finlang/voting_72h.log 2>&1

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
./run.sh voting_72h

# 6. Setup crontab
(crontab -l 2>/dev/null; echo "5 * * * * /opt/finlang/run.sh voting_72h >> /var/log/finlang/voting_72h.log 2>&1") | crontab -
```

### Verify Deployment

```bash
# Check crontab
crontab -l

# Monitor logs
tail -f /var/log/finlang/voting_72h.log

# Check latest signal
cat /opt/finlang/signals/btcusdt_voting_72h_latest.json
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
  "strategy": "voting_72h",
  "source": "prediction_api",
  "factors": {
    "sig_24h": 65.0,
    "history_count": 72,
    "bullish_pct": 58.3,
    "bullish_count": 42,
    "total_count": 72,
    "allow_short": true
  }
}
```

### Position Values

| Position | Action | Meaning |
|----------|--------|---------|
| `1` | LONG | Buy / Hold long |
| `0` | FLAT | No position / Exit |
| `-1` | SHORT | Sell short (voting_72h only) |

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
│   └── btcusdt_voting_72h_latest.json
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
