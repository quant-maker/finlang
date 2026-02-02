#!/usr/bin/env python3
"""
Unified Cron Script - All strategies using remote prediction API.

Supports:
- Voting 48h strategy (48h window, 58% threshold, Sharpe 1.45, BEATS B&H - recommended)
- Voting strategy (96h window, 50% threshold, Sharpe ~0.65)
- Voting 120h strategy (120h window, 58% threshold, Sharpe ~0.97)
- MLP strategy (online learning neural network)
- Dynamic strategy (adaptive thresholds)

All strategies use predictions from remote prediction API.
Cold start: Automatically backfills missing history using custom data API.

Dependencies: requests, numpy, pandas

Configuration:
- Set FINLANG_API_URL environment variable to specify the prediction service URL

Usage:
    # Run with recommended strategy (voting_48h - beats B&H)
    python -m finlang.cron --symbol BTCUSDT --strategy voting_48h
    
    # Run with other strategies
    python -m finlang.cron --symbol BTCUSDT --strategy voting
    python -m finlang.cron --symbol BTCUSDT --strategy voting_120h
    python -m finlang.cron --symbol BTCUSDT --strategy mlp
    python -m finlang.cron --symbol BTCUSDT --strategy dynamic
    
    # Backfill history (cold start)
    python -m finlang.cron --symbol BTCUSDT --backfill 300
    
    # Force retrain MLP
    python -m finlang.cron --symbol BTCUSDT --strategy mlp --force-retrain

Crontab:
    5 * * * * cd /path/to/finlang && python -m finlang.cron --symbol BTCUSDT --strategy voting_48h >> /var/log/finlang.log 2>&1
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

# Import API client
from finlang.api import (
    get_latest_prediction,
    get_prediction_for_time,
    fetch_binance_ohlcv,
    DEFAULT_API_URL,
)
from finlang.model.mlp import OnlineMLP


# =============================================================================
# Configuration
# =============================================================================

# Voting strategy configurations (based on backtest with 25928 hours of data, 72h rebalance)
# voting_48h: 48h window, long_threshold=0.58, Sharpe 1.45 (BEATS B&H, recommended)
# voting: 96h window, long_threshold=0.50, Sharpe ~0.65 (underperforms B&H)
# voting_120h: 120h window, long_threshold=0.58, Sharpe ~0.97 (underperforms B&H)
VOTING_CONFIGS = {
    "voting_48h": {
        "window": 48,
        "long_threshold": 0.58,
        "short_threshold": 0.42,  # flat threshold (no short)
        "allow_short": False,
    },
    "voting": {
        "window": 96,
        "long_threshold": 0.50,
        "short_threshold": 0.40,
        "allow_short": False,
    },
    "voting_120h": {
        "window": 120,
        "long_threshold": 0.58,
        "short_threshold": 0.40,
        "allow_short": False,
    },
}

# MLP parameters
MLP_LOOKAHEAD = 72           # Hours to look ahead for labels
MLP_TRAIN_MIN = 200          # Minimum samples for training
MLP_RETRAIN_FREQ = 168       # Retrain every week
MLP_HIDDEN_DIM = 16
MLP_EPOCHS = 50

# Dynamic strategy parameters
DYNAMIC_BULL_THRESHOLD = 0.60
DYNAMIC_BEAR_THRESHOLD = 0.40
DYNAMIC_WINDOW = 48

# Backfill settings
BACKFILL_BATCH_SIZE = 10     # Predictions per batch
BACKFILL_DELAY = 2           # Seconds between API calls


# =============================================================================
# History Management
# =============================================================================

def get_history_dir(base_dir: Path, symbol: str) -> Path:
    """Get history directory for a symbol."""
    hist_dir = base_dir / "history" / symbol.lower()
    hist_dir.mkdir(parents=True, exist_ok=True)
    return hist_dir


def get_model_dir(base_dir: Path) -> Path:
    """Get model directory."""
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_prediction(pred: Dict[str, Any], hist_dir: Path) -> Path:
    """Save prediction to history file."""
    data_ts = pred.get("data_timestamp") or pred.get("timestamp")
    if data_ts:
        if isinstance(data_ts, str):
            ts = datetime.fromisoformat(data_ts.replace("Z", "+00:00"))
        else:
            ts = data_ts
    else:
        ts = datetime.now(timezone.utc)
    
    ts_aligned = ts.replace(minute=0, second=0, microsecond=0)
    filename = ts_aligned.strftime("%Y%m%d_%H00.json")
    filepath = hist_dir / filename
    
    record = {
        "timestamp": ts_aligned.isoformat(),
        "sig_24h": pred.get("sig_24h", pred.get("upside_probability", 0.5)),
        "current_close": pred.get("current_close", 0),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "source": pred.get("source", "prediction_api"),
    }
    
    with open(filepath, "w") as f:
        json.dump(record, f, indent=2, default=str)
    
    return filepath


def load_history(hist_dir: Path, hours: int = 0) -> List[Dict[str, Any]]:
    """Load prediction history sorted by timestamp."""
    if not hist_dir.exists():
        return []
    
    files = sorted(hist_dir.glob("*.json"))
    
    if hours > 0 and len(files) > hours:
        files = files[-hours:]
    
    records = []
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
                records.append(data)
        except Exception as e:
            continue
    
    return records


def get_missing_hours(hist_dir: Path, required_hours: int) -> List[datetime]:
    """
    Find missing hours in history that need to be backfilled.
    
    Returns list of datetime objects for missing hours.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    
    # Get existing timestamps
    existing = set()
    for f in hist_dir.glob("*.json"):
        # Parse timestamp from filename: YYYYMMDD_HH00.json
        try:
            ts_str = f.stem  # e.g., "20260131_1200"
            ts = datetime.strptime(ts_str, "%Y%m%d_%H00").replace(tzinfo=timezone.utc)
            existing.add(ts)
        except:
            continue
    
    # Find missing hours
    missing = []
    for i in range(required_hours):
        check_time = now - timedelta(hours=i)
        if check_time not in existing:
            missing.append(check_time)
    
    return sorted(missing)


def backfill_missing(
    hist_dir: Path,
    symbol: str,
    required_hours: int,
    api_url: str = DEFAULT_API_URL,
) -> int:
    """
    Backfill missing predictions using custom data API.
    
    Returns number of predictions backfilled.
    """
    import time
    
    missing = get_missing_hours(hist_dir, required_hours)
    
    if not missing:
        return 0
    
    print(f"[BACKFILL] Found {len(missing)} missing hours, backfilling...")
    
    count = 0
    for i, target_time in enumerate(missing):
        try:
            print(f"[BACKFILL] {i+1}/{len(missing)}: {target_time.strftime('%Y-%m-%d %H:00')}")
            
            pred = get_prediction_for_time(
                target_time=target_time,
                symbol=symbol,
                hist_bars=384,
                api_url=api_url,
            )
            
            save_prediction(pred, hist_dir)
            count += 1
            
            # Rate limiting
            time.sleep(BACKFILL_DELAY)
            
        except Exception as e:
            print(f"[BACKFILL] Failed for {target_time}: {e}")
            continue
    
    print(f"[BACKFILL] Completed: {count}/{len(missing)} predictions")
    return count


# =============================================================================
# Strategies
# =============================================================================

def voting_strategy(
    records: List[Dict],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Voting strategy: Count bullish votes in rolling window.
    
    Args:
        records: List of historical prediction records
        config: Strategy configuration dict with keys:
            - window: Number of hours to look back
            - long_threshold: Bullish % to go long
            - short_threshold: Bullish % to go short (or flat if allow_short=False)
            - allow_short: Whether to allow short positions
    """
    # Default to standard voting config
    if config is None:
        config = VOTING_CONFIGS["voting"]
    
    window = config["window"]
    long_threshold = config["long_threshold"]
    short_threshold = config["short_threshold"]
    allow_short = config.get("allow_short", False)
    
    if not records:
        return {"error": "No predictions available"}
    
    window_records = records[-window:]
    
    bullish_count = 0
    for rec in window_records:
        sig_24h = rec.get("sig_24h", 0.5)
        if isinstance(sig_24h, str):
            sig_24h = float(sig_24h)
        if sig_24h > 1:
            sig_24h = sig_24h / 100
        if sig_24h > 0.5:
            bullish_count += 1
    
    total_count = len(window_records)
    bullish_pct = bullish_count / total_count if total_count > 0 else 0.5
    
    # Decision logic with short support
    if bullish_pct >= long_threshold:
        position, action, trend = 1, "LONG", "bullish"
    elif allow_short and bullish_pct < short_threshold:
        position, action, trend = -1, "SHORT", "bearish"
    elif bullish_pct < short_threshold:
        position, action, trend = 0, "FLAT", "bearish"
    else:
        position, action, trend = 0, "FLAT", "neutral"
    
    return {
        "position": position,
        "action": action,
        "trend": trend,
        "score": bullish_pct,
        "bullish_pct": bullish_pct,
        "bullish_count": bullish_count,
        "total_count": total_count,
        "allow_short": allow_short,
    }


def mlp_strategy(
    records: List[Dict],
    model_path: Path,
    force_retrain: bool = False,
    current_sig_24h: float = 0.5,
) -> Dict[str, Any]:
    """
    MLP strategy: Online learning neural network.
    """
    # Load or create model
    mlp = OnlineMLP(
        input_dim=3,
        hidden_dim=MLP_HIDDEN_DIM,
        output_dim=1,
        lr=0.01,
    )
    
    if model_path.exists():
        mlp.load_model(str(model_path))
    
    # Check if we need to train
    should_train = False
    if not mlp.trained:
        should_train = True
    elif force_retrain:
        should_train = True
    elif model_path.exists():
        model_age = (datetime.now(timezone.utc) - 
                    datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc))
        if model_age.total_seconds() / 3600 >= MLP_RETRAIN_FREQ:
            should_train = True
    
    train_stats = None
    if should_train and len(records) >= MLP_TRAIN_MIN + MLP_LOOKAHEAD:
        train_stats = _train_mlp(mlp, records, model_path)
    
    # Generate signal
    if mlp.trained:
        features = np.array([current_sig_24h, current_sig_24h**2, current_sig_24h - 0.5])
        prob = float(mlp.predict(features.reshape(1, -1))[0, 0])
        
        if prob >= 0.55:
            position, action, trend = 1, "LONG", "bullish"
        elif prob < 0.45:
            position, action, trend = 0, "FLAT", "bearish"
        else:
            position, action, trend = 0, "FLAT", "neutral"
        
        return {
            "position": position,
            "action": action,
            "trend": trend,
            "score": prob,
            "mlp_prob": prob,
            "mlp_trained": True,
            "train_stats": train_stats,
        }
    else:
        # Fallback to voting
        return voting_strategy(records)


def _train_mlp(mlp: OnlineMLP, records: List[Dict], model_path: Path) -> Dict[str, Any]:
    """Train MLP on historical data."""
    X_list, y_list = [], []
    
    n = len(records)
    for i in range(n - MLP_LOOKAHEAD):
        current = records[i]
        future = records[i + MLP_LOOKAHEAD]
        
        sig_24h = current.get("sig_24h", 0.5)
        if isinstance(sig_24h, str):
            sig_24h = float(sig_24h)
        if sig_24h > 1:
            sig_24h = sig_24h / 100
        
        current_price = current.get("current_close", 0)
        future_price = future.get("current_close", 0)
        
        if current_price <= 0 or future_price <= 0:
            continue
        
        features = np.array([sig_24h, sig_24h**2, sig_24h - 0.5])
        label = 1.0 if future_price > current_price else 0.0
        
        X_list.append(features)
        y_list.append(label)
    
    if len(X_list) < MLP_TRAIN_MIN:
        return {"trained": False, "reason": f"Not enough samples: {len(X_list)}"}
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    loss = mlp.fit(X, y, epochs=MLP_EPOCHS, batch_size=32)
    mlp.save_model(str(model_path))
    
    accuracy = np.mean((mlp.predict(X).flatten() > 0.5) == y)
    
    return {
        "trained": True,
        "samples": len(X),
        "loss": float(loss),
        "accuracy": float(accuracy),
    }


def dynamic_strategy(
    records: List[Dict],
    window: int = DYNAMIC_WINDOW,
) -> Dict[str, Any]:
    """
    Dynamic strategy: Adaptive thresholds based on recent volatility.
    """
    if len(records) < window:
        return voting_strategy(records)
    
    window_records = records[-window:]
    
    # Calculate recent volatility
    prices = [r.get("current_close", 0) for r in window_records if r.get("current_close", 0) > 0]
    if len(prices) < 10:
        return voting_strategy(records)
    
    prices = np.array(prices)
    returns = np.diff(np.log(prices))
    volatility = np.std(returns) * np.sqrt(24)  # Annualized hourly vol
    
    # Adaptive thresholds
    if volatility > 0.03:  # High vol
        long_threshold = 0.65
        flat_threshold = 0.35
    elif volatility < 0.015:  # Low vol
        long_threshold = 0.52
        flat_threshold = 0.48
    else:  # Normal vol
        long_threshold = 0.58
        flat_threshold = 0.42
    
    # Count bullish signals
    bullish_count = 0
    for rec in window_records:
        sig_24h = rec.get("sig_24h", 0.5)
        if isinstance(sig_24h, str):
            sig_24h = float(sig_24h)
        if sig_24h > 1:
            sig_24h = sig_24h / 100
        if sig_24h > 0.5:
            bullish_count += 1
    
    bullish_pct = bullish_count / len(window_records)
    
    if bullish_pct >= long_threshold:
        position, action, trend = 1, "LONG", "bullish"
    elif bullish_pct < flat_threshold:
        position, action, trend = 0, "FLAT", "bearish"
    else:
        position, action, trend = 0, "FLAT", "neutral"
    
    return {
        "position": position,
        "action": action,
        "trend": trend,
        "score": bullish_pct,
        "bullish_pct": bullish_pct,
        "volatility": volatility,
        "long_threshold": long_threshold,
        "flat_threshold": flat_threshold,
    }


# =============================================================================
# Main Logic
# =============================================================================

def send_webhook(url: str, signal: Dict[str, Any]) -> bool:
    """Send signal to webhook URL."""
    import requests
    try:
        response = requests.post(url, json=signal, headers={"Content-Type": "application/json"}, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"[ERROR] Webhook failed: {e}")
        return False


def run(
    symbol: str = "BTCUSDT",
    strategy: str = "voting",
    api_url: str = DEFAULT_API_URL,
    base_dir: Optional[str] = None,
    signal_dir: Optional[str] = None,
    webhook_url: Optional[str] = None,
    force_retrain: bool = False,
    save_signal: bool = True,
    auto_backfill: bool = True,
) -> Dict[str, Any]:
    """
    Main entry point.
    """
    now = datetime.now(timezone.utc)
    print(f"[{now.isoformat()}] Starting {strategy} prediction for {symbol}")
    
    # Setup paths
    if base_dir is None:
        base_dir = Path(__file__).parent
    else:
        base_dir = Path(base_dir)
    
    if signal_dir is None:
        signal_dir = base_dir / "signals"
    else:
        signal_dir = Path(signal_dir)
    
    signal_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = get_history_dir(base_dir, symbol)
    model_dir = get_model_dir(base_dir)
    
    # Step 1: Fetch current prediction
    print(f"[INFO] Fetching prediction from API...")
    try:
        pred = get_latest_prediction(api_url=api_url, simple=True)
        print(f"[INFO] Got prediction: sig_24h={pred['sig_24h']:.1%}, price=${pred['current_close']:,.2f}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch prediction: {e}")
        return {"timestamp": now.isoformat(), "symbol": symbol, "position": 0, "action": "ERROR", "error": str(e)}
    
    pred["data_timestamp"] = now.isoformat()
    current_close = pred["current_close"]
    current_sig_24h = pred["sig_24h"]
    
    # Step 2: Save to history
    save_prediction(pred, hist_dir)
    
    # Step 3: Load history
    records = load_history(hist_dir)
    print(f"[INFO] Loaded {len(records)} historical records")
    
    # Step 4: Auto-backfill if needed
    # Get required hours based on strategy
    if strategy in VOTING_CONFIGS:
        required_hours = VOTING_CONFIGS[strategy]["window"]
    elif strategy == "mlp":
        required_hours = MLP_TRAIN_MIN + MLP_LOOKAHEAD
    elif strategy == "dynamic":
        required_hours = DYNAMIC_WINDOW
    else:
        required_hours = VOTING_CONFIGS["voting"]["window"]
    
    if auto_backfill and len(records) < required_hours:
        print(f"[INFO] History insufficient ({len(records)}/{required_hours}), backfilling...")
        backfill_missing(hist_dir, symbol, required_hours, api_url)
        records = load_history(hist_dir)
        print(f"[INFO] After backfill: {len(records)} records")
    
    # Step 5: Run strategy
    if strategy == "mlp":
        model_path = model_dir / f"mlp_{symbol.lower()}.npz"
        result = mlp_strategy(records, model_path, force_retrain, current_sig_24h)
    elif strategy == "dynamic":
        result = dynamic_strategy(records)
    elif strategy in VOTING_CONFIGS:
        # voting or voting_72h
        result = voting_strategy(records, config=VOTING_CONFIGS[strategy])
    else:
        # Fallback to standard voting
        result = voting_strategy(records)
    
    if "error" in result:
        return {"timestamp": now.isoformat(), "symbol": symbol, "position": 0, "action": "ERROR", "error": result["error"]}
    
    # Build final signal
    signal = {
        "timestamp": now.isoformat(),
        "symbol": symbol,
        "position": result["position"],
        "action": result["action"],
        "trend": result["trend"],
        "score": result.get("score", 0.5),
        "current_close": current_close,
        "strategy": strategy,
        "source": "prediction_api",
        "factors": {
            "sig_24h": round(current_sig_24h * 100, 1),
            "history_count": len(records),
        },
    }
    
    # Add strategy-specific factors
    if strategy == "mlp":
        signal["factors"]["mlp_prob"] = round(result.get("mlp_prob", 0) * 100, 1)
        signal["factors"]["mlp_trained"] = result.get("mlp_trained", False)
    elif strategy == "dynamic":
        signal["factors"]["volatility"] = round(result.get("volatility", 0) * 100, 2)
        signal["factors"]["long_threshold"] = result.get("long_threshold", 0)
    elif strategy in VOTING_CONFIGS:
        # voting or voting_72h
        signal["factors"]["bullish_pct"] = round(result.get("bullish_pct", 0) * 100, 1)
        signal["factors"]["bullish_count"] = result.get("bullish_count", 0)
        signal["factors"]["total_count"] = result.get("total_count", 0)
        signal["factors"]["allow_short"] = result.get("allow_short", False)
    else:
        signal["factors"]["bullish_pct"] = round(result.get("bullish_pct", 0) * 100, 1)
        signal["factors"]["bullish_count"] = result.get("bullish_count", 0)
        signal["factors"]["total_count"] = result.get("total_count", 0)
    
    print(f"[SIGNAL] {signal['symbol']}: {signal['action']} (strategy={strategy})")
    print(f"[SIGNAL] Price: ${signal['current_close']:,.2f}")
    
    # Step 6: Save signal
    if save_signal:
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        signal_file = signal_dir / f"{symbol.lower()}_{strategy}_{timestamp}.json"
        with open(signal_file, "w") as f:
            json.dump(signal, f, indent=2, default=str)
        
        latest_file = signal_dir / f"{symbol.lower()}_{strategy}_latest.json"
        with open(latest_file, "w") as f:
            json.dump(signal, f, indent=2, default=str)
    
    # Send to webhook
    if webhook_url:
        send_webhook(webhook_url, signal)
    
    return signal


def main():
    parser = argparse.ArgumentParser(description="Unified cron signal generator with all strategies")
    parser.add_argument("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--strategy", "-t", choices=["voting_48h", "voting", "voting_120h", "mlp", "dynamic"], default="voting_48h", help="Strategy to use (voting_48h recommended)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Prediction API URL")
    parser.add_argument("--signal-dir", help="Signal output directory")
    parser.add_argument("--webhook", help="Webhook URL")
    parser.add_argument("--force-retrain", action="store_true", help="Force MLP retraining")
    parser.add_argument("--no-save", action="store_true", help="Don't save signal")
    parser.add_argument("--no-backfill", action="store_true", help="Disable auto-backfill")
    parser.add_argument("--backfill", type=int, metavar="HOURS", help="Manually backfill N hours")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    # Handle manual backfill mode
    if args.backfill:
        base_dir = Path(__file__).parent
        hist_dir = get_history_dir(base_dir, args.symbol)
        print(f"[BACKFILL] Backfilling {args.backfill} hours for {args.symbol}...")
        count = backfill_missing(hist_dir, args.symbol, args.backfill, args.api_url or DEFAULT_API_URL)
        print(f"\n[DONE] Backfilled {count} predictions")
        return
    
    # Normal run
    api_url = args.api_url or os.environ.get("FINLANG_API_URL", DEFAULT_API_URL)
    webhook_url = args.webhook or os.environ.get("FINLANG_WEBHOOK_URL")
    signal_dir = args.signal_dir or os.environ.get("FINLANG_SIGNAL_DIR")
    
    signal = run(
        symbol=args.symbol,
        strategy=args.strategy,
        api_url=api_url,
        signal_dir=signal_dir,
        webhook_url=webhook_url,
        force_retrain=args.force_retrain,
        save_signal=not args.no_save,
        auto_backfill=not args.no_backfill,
    )
    
    if args.json:
        print(json.dumps(signal, indent=2, default=str))
    else:
        factors = signal.get("factors", {})
        print(f"\n{'='*50}")
        print(f"SIGNAL: {signal['symbol']} ({signal['strategy']})")
        print(f"  Action: {signal['action']}")
        print(f"  Position: {signal['position']:+d}")
        print(f"  sig_24h: {factors.get('sig_24h', 0):.1f}%")
        
        if "mlp_prob" in factors:
            print(f"  MLP Prob: {factors['mlp_prob']:.1f}%")
        if "bullish_pct" in factors:
            print(f"  Bullish: {factors['bullish_pct']:.1f}%")
        if "volatility" in factors:
            print(f"  Volatility: {factors['volatility']:.2f}%")
        
        print(f"  History: {factors.get('history_count', 0)} records")
        print(f"  Price: ${signal['current_close']:,.2f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
