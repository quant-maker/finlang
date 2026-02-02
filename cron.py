#!/usr/bin/env python3
"""
Unified Cron Script - Config-based strategy execution.

Each strategy instance is identified by --name, which determines:
- config/{name}.json - Strategy configuration (type, params, symbol)
- state/{name}.json  - Runtime state (position, last_rebalance_time)

Supports:
- voting: Voting strategy with configurable window/threshold
- mlp: Online learning neural network
- dynamic: Adaptive thresholds based on volatility

IMPORTANT: Rebalance frequency is critical for voting strategy!
    - 1h rebalance  = -3% return (loses badly to B&H)
    - 72h rebalance = 275% return (beats B&H with Sharpe 1.45)

Usage:
    # Run a named strategy (config from config/btc_voting.json)
    python -m finlang.cron --name btc_voting
    
    # Initialize a new config
    python -m finlang.cron --init voting --name btc_voting
    
    # Backfill history for a symbol
    python -m finlang.cron --backfill 300 --symbol BTCUSDT

Config Example (config/btc_voting.json):
    {
        "strategy": "voting",
        "symbol": "BTCUSDT",
        "window": 48,
        "threshold": 0.58,
        "rebalance_hours": 72
    }

Crontab (run every hour, but only rebalance per config):
    5 * * * * cd /path/to/finlang && python -m finlang.cron --name btc_voting >> /var/log/finlang.log 2>&1
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
# Default Configuration Values
# =============================================================================

# Voting strategy defaults (based on backtest: beats B&H with Sharpe 1.45)
DEFAULT_VOTING_CONFIG = {
    "strategy": "voting",
    "symbol": "BTCUSDT",
    "window": 48,
    "threshold": 0.58,          # bullish_pct >= threshold -> LONG
    "short_threshold": 0.42,    # bullish_pct <= short_threshold -> SHORT
    "allow_short": True,        # enable short positions
    "rebalance_hours": 72,
}

# MLP strategy defaults
DEFAULT_MLP_CONFIG = {
    "strategy": "mlp",
    "symbol": "BTCUSDT",
    "lookahead": 72,
    "train_min": 200,
    "retrain_freq": 168,
    "hidden_dim": 16,
    "epochs": 50,
    "long_threshold": 0.55,     # prob >= long_threshold -> LONG
    "short_threshold": 0.45,    # prob <= short_threshold -> SHORT
    "allow_short": True,
    "rebalance_hours": 24,
}

# Dynamic strategy defaults
DEFAULT_DYNAMIC_CONFIG = {
    "strategy": "dynamic",
    "symbol": "BTCUSDT",
    "window": 48,
    "allow_short": True,
    "rebalance_hours": 24,
}

# Backfill settings
BACKFILL_DELAY = 2  # Seconds between API calls


# =============================================================================
# Config Management
# =============================================================================

def get_config_path(base_dir: Path, name: str) -> Path:
    """Get config file path for a named strategy."""
    return base_dir / "config" / f"{name}.json"


def get_state_path(base_dir: Path, name: str) -> Path:
    """Get state file path for a named strategy."""
    return base_dir / "state" / f"{name}.json"


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load config from file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}\nRun with --init to create one.")
    with open(config_path) as f:
        return json.load(f)


def save_config(config_path: Path, config: Dict[str, Any]) -> None:
    """Save config to file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def load_state(state_path: Path) -> Dict[str, Any]:
    """Load state from file."""
    if not state_path.exists():
        return {}
    try:
        with open(state_path) as f:
            return json.load(f)
    except:
        return {}


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    """Save state to file."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def should_rebalance(state: Dict[str, Any], rebalance_hours: int) -> bool:
    """
    Check if we should rebalance based on last rebalance time.
    
    Returns True if:
    - No previous state (first run)
    - Last rebalance was >= rebalance_hours ago
    
    Note: Times are rounded to the hour to avoid edge cases where
    a few seconds/minutes difference causes an extra hour wait.
    (e.g., cron runs at 5 * * * * but program takes a few seconds)
    """
    last_rebalance = state.get("last_rebalance_time")
    if not last_rebalance:
        return True
    
    try:
        last_dt = datetime.fromisoformat(last_rebalance.replace("Z", "+00:00"))
        # Round both times to the hour to avoid edge cases
        last_hour = last_dt.replace(minute=0, second=0, microsecond=0)
        now_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        hours_since = (now_hour - last_hour).total_seconds() / 3600
        return hours_since >= rebalance_hours
    except:
        return True


def init_config(base_dir: Path, name: str, strategy: str) -> Path:
    """Initialize a new config file with defaults."""
    config_path = get_config_path(base_dir, name)
    
    if config_path.exists():
        print(f"[WARN] Config already exists: {config_path}")
        return config_path
    
    if strategy == "voting":
        config = DEFAULT_VOTING_CONFIG.copy()
    elif strategy == "mlp":
        config = DEFAULT_MLP_CONFIG.copy()
    elif strategy == "dynamic":
        config = DEFAULT_DYNAMIC_CONFIG.copy()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    save_config(config_path, config)
    print(f"[OK] Created config: {config_path}")
    print(json.dumps(config, indent=2))
    return config_path


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
    """Find missing hours in history that need to be backfilled."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    
    existing = set()
    for f in hist_dir.glob("*.json"):
        try:
            ts_str = f.stem
            ts = datetime.strptime(ts_str, "%Y%m%d_%H00").replace(tzinfo=timezone.utc)
            existing.add(ts)
        except:
            continue
    
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
    """Backfill missing predictions using custom data API."""
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
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Voting strategy: Count bullish votes in rolling window.
    
    Config params:
        window: Number of hours to look back (default: 48)
        threshold: Bullish % to go long (default: 0.58)
        short_threshold: Bullish % to go short (default: 0.42)
        allow_short: Enable short positions (default: True)
    """
    window = config.get("window", 48)
    threshold = config.get("threshold", 0.58)
    short_threshold = config.get("short_threshold", 0.42)
    allow_short = config.get("allow_short", True)
    
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
    
    # Decision logic
    if bullish_pct >= threshold:
        position, action, trend = 1, "LONG", "bullish"
    elif allow_short and bullish_pct <= short_threshold:
        position, action, trend = -1, "SHORT", "bearish"
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
        "window": window,
        "threshold": threshold,
        "short_threshold": short_threshold,
        "allow_short": allow_short,
    }


def mlp_strategy(
    records: List[Dict],
    config: Dict[str, Any],
    model_path: Path,
    force_retrain: bool = False,
    current_sig_24h: float = 0.5,
) -> Dict[str, Any]:
    """MLP strategy: Online learning neural network."""
    lookahead = config.get("lookahead", 72)
    train_min = config.get("train_min", 200)
    retrain_freq = config.get("retrain_freq", 168)
    hidden_dim = config.get("hidden_dim", 16)
    epochs = config.get("epochs", 50)
    long_threshold = config.get("long_threshold", 0.55)
    short_threshold = config.get("short_threshold", 0.45)
    allow_short = config.get("allow_short", True)
    
    mlp = OnlineMLP(
        input_dim=3,
        hidden_dim=hidden_dim,
        output_dim=1,
        lr=0.01,
    )
    
    if model_path.exists():
        mlp.load_model(str(model_path))
    
    should_train = False
    if not mlp.trained:
        should_train = True
    elif force_retrain:
        should_train = True
    elif model_path.exists():
        model_age = (datetime.now(timezone.utc) - 
                    datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc))
        if model_age.total_seconds() / 3600 >= retrain_freq:
            should_train = True
    
    train_stats = None
    if should_train and len(records) >= train_min + lookahead:
        train_stats = _train_mlp(mlp, records, model_path, lookahead, train_min, epochs)
    
    if mlp.trained:
        features = np.array([current_sig_24h, current_sig_24h**2, current_sig_24h - 0.5])
        prob = float(mlp.predict(features.reshape(1, -1))[0, 0])
        
        if prob >= long_threshold:
            position, action, trend = 1, "LONG", "bullish"
        elif allow_short and prob <= short_threshold:
            position, action, trend = -1, "SHORT", "bearish"
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
            "long_threshold": long_threshold,
            "short_threshold": short_threshold,
            "allow_short": allow_short,
        }
    else:
        return voting_strategy(records, {"window": 48, "threshold": 0.58, "allow_short": allow_short})


def _train_mlp(
    mlp: OnlineMLP, 
    records: List[Dict], 
    model_path: Path,
    lookahead: int,
    train_min: int,
    epochs: int,
) -> Dict[str, Any]:
    """Train MLP on historical data."""
    X_list, y_list = [], []
    
    n = len(records)
    for i in range(n - lookahead):
        current = records[i]
        future = records[i + lookahead]
        
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
    
    if len(X_list) < train_min:
        return {"trained": False, "reason": f"Not enough samples: {len(X_list)}"}
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    loss = mlp.fit(X, y, epochs=epochs, batch_size=32)
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
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Dynamic strategy: Adaptive thresholds based on recent volatility."""
    window = config.get("window", 48)
    allow_short = config.get("allow_short", True)
    
    if len(records) < window:
        return voting_strategy(records, {"window": window, "threshold": 0.58, "allow_short": allow_short})
    
    window_records = records[-window:]
    
    prices = [r.get("current_close", 0) for r in window_records if r.get("current_close", 0) > 0]
    if len(prices) < 10:
        return voting_strategy(records, {"window": window, "threshold": 0.58, "allow_short": allow_short})
    
    prices = np.array(prices)
    returns = np.diff(np.log(prices))
    volatility = np.std(returns) * np.sqrt(24)
    
    # Adaptive thresholds based on volatility
    if volatility > 0.03:       # High vol: be more selective
        long_threshold = 0.65
        short_threshold = 0.35
    elif volatility < 0.015:    # Low vol: be more aggressive
        long_threshold = 0.52
        short_threshold = 0.48
    else:                       # Normal vol
        long_threshold = 0.58
        short_threshold = 0.42
    
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
    elif allow_short and bullish_pct <= short_threshold:
        position, action, trend = -1, "SHORT", "bearish"
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
        "short_threshold": short_threshold,
        "allow_short": allow_short,
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
    name: str,
    api_url: str = DEFAULT_API_URL,
    base_dir: Optional[Path] = None,
    signal_dir: Optional[Path] = None,
    webhook_url: Optional[str] = None,
    force_retrain: bool = False,
    save_signal: bool = True,
    auto_backfill: bool = True,
) -> Dict[str, Any]:
    """
    Main entry point - run a named strategy.
    
    Args:
        name: Strategy instance name (loads config/{name}.json)
        api_url: Prediction API URL
        base_dir: Base directory for config/state/history
        ...
    """
    now = datetime.now(timezone.utc)
    
    # Setup paths
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    if signal_dir is None:
        signal_dir = base_dir / "signals"
    
    signal_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config_path = get_config_path(base_dir, name)
    config = load_config(config_path)
    
    strategy = config.get("strategy", "voting")
    symbol = config.get("symbol", "BTCUSDT")
    rebalance_hours = config.get("rebalance_hours", 72)
    
    print(f"[{now.isoformat()}] Running '{name}' ({strategy}) for {symbol}")
    
    # Setup paths
    hist_dir = get_history_dir(base_dir, symbol)
    model_dir = get_model_dir(base_dir)
    state_path = get_state_path(base_dir, name)
    state = load_state(state_path)
    
    # Check if we should rebalance
    if rebalance_hours > 1 and not should_rebalance(state, rebalance_hours):
        now_hour = now.replace(minute=0, second=0, microsecond=0)
        hours_since = 0
        if state.get("last_rebalance_time"):
            try:
                last_dt = datetime.fromisoformat(state["last_rebalance_time"].replace("Z", "+00:00"))
                last_hour = last_dt.replace(minute=0, second=0, microsecond=0)
                hours_since = (now_hour - last_hour).total_seconds() / 3600
            except:
                pass
        
        print(f"[INFO] Not time to rebalance yet ({hours_since:.0f}h / {rebalance_hours}h)")
        print(f"[INFO] Holding position: {state.get('action', 'UNKNOWN')}")
        
        return {
            "timestamp": now.isoformat(),
            "name": name,
            "symbol": symbol,
            "strategy": strategy,
            "position": state.get("position", 0),
            "action": state.get("action", "HOLD"),
            "trend": state.get("trend", "unknown"),
            "score": state.get("score", 0.5),
            "current_close": state.get("entry_price", 0),
            "source": "state_cache",
            "rebalance": False,
            "hours_until_rebalance": int(rebalance_hours - hours_since),
            "config": config,
        }
    
    # Fetch current prediction
    print(f"[INFO] Fetching prediction from API...")
    try:
        pred = get_latest_prediction(api_url=api_url, simple=True)
        print(f"[INFO] Got prediction: sig_24h={pred['sig_24h']:.1%}, price=${pred['current_close']:,.2f}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch prediction: {e}")
        return {"timestamp": now.isoformat(), "name": name, "symbol": symbol, "position": 0, "action": "ERROR", "error": str(e)}
    
    pred["data_timestamp"] = now.isoformat()
    current_close = pred["current_close"]
    current_sig_24h = pred["sig_24h"]
    
    # Save to history
    save_prediction(pred, hist_dir)
    
    # Load history
    records = load_history(hist_dir)
    print(f"[INFO] Loaded {len(records)} historical records")
    
    # Determine required history
    if strategy == "voting":
        required_hours = config.get("window", 48)
    elif strategy == "mlp":
        required_hours = config.get("train_min", 200) + config.get("lookahead", 72)
    elif strategy == "dynamic":
        required_hours = config.get("window", 48)
    else:
        required_hours = 48
    
    # Auto-backfill if needed
    if auto_backfill and len(records) < required_hours:
        print(f"[INFO] History insufficient ({len(records)}/{required_hours}), backfilling...")
        backfill_missing(hist_dir, symbol, required_hours, api_url)
        records = load_history(hist_dir)
        print(f"[INFO] After backfill: {len(records)} records")
    
    # Run strategy
    if strategy == "mlp":
        model_path = model_dir / f"mlp_{name}.npz"
        result = mlp_strategy(records, config, model_path, force_retrain, current_sig_24h)
    elif strategy == "dynamic":
        result = dynamic_strategy(records, config)
    else:
        result = voting_strategy(records, config)
    
    if "error" in result:
        return {"timestamp": now.isoformat(), "name": name, "symbol": symbol, "position": 0, "action": "ERROR", "error": result["error"]}
    
    # Build signal
    signal = {
        "timestamp": now.isoformat(),
        "name": name,
        "symbol": symbol,
        "strategy": strategy,
        "position": result["position"],
        "action": result["action"],
        "trend": result["trend"],
        "score": result.get("score", 0.5),
        "current_close": current_close,
        "source": "prediction_api",
        "rebalance": True,
        "config": config,
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
    else:
        signal["factors"]["bullish_pct"] = round(result.get("bullish_pct", 0) * 100, 1)
        signal["factors"]["bullish_count"] = result.get("bullish_count", 0)
        signal["factors"]["total_count"] = result.get("total_count", 0)
    
    # Save state
    new_state = {
        "last_rebalance_time": now.isoformat(),
        "position": result["position"],
        "action": result["action"],
        "trend": result["trend"],
        "score": result.get("score", 0.5),
        "entry_price": current_close,
        "last_sig_24h": round(current_sig_24h * 100, 1),
        "history_count": len(records),
    }
    save_state(state_path, new_state)
    print(f"[INFO] Rebalanced! Saved state, next rebalance in {rebalance_hours}h")
    
    print(f"[SIGNAL] {name}: {signal['action']} ({strategy})")
    print(f"[SIGNAL] Price: ${signal['current_close']:,.2f}")
    
    # Save signal
    if save_signal:
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        signal_file = signal_dir / f"{name}_{timestamp}.json"
        with open(signal_file, "w") as f:
            json.dump(signal, f, indent=2, default=str)
        
        latest_file = signal_dir / f"{name}_latest.json"
        with open(latest_file, "w") as f:
            json.dump(signal, f, indent=2, default=str)
    
    # Send to webhook
    if webhook_url:
        send_webhook(webhook_url, signal)
    
    return signal


def main():
    parser = argparse.ArgumentParser(
        description="Config-based strategy execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a named strategy (config from config/btc_voting.json)
  python -m finlang.cron --name btc_voting
  
  # Initialize a new voting config
  python -m finlang.cron --init voting --name btc_voting
  
  # Initialize a new mlp config
  python -m finlang.cron --init mlp --name btc_mlp
  
  # Backfill history for a symbol
  python -m finlang.cron --backfill 300 --symbol BTCUSDT

Config files are stored in config/{name}.json
State files are stored in state/{name}.json
        """
    )
    parser.add_argument("--name", "-n", help="Strategy instance name (required for run)")
    parser.add_argument("--init", choices=["voting", "mlp", "dynamic"], help="Initialize a new config")
    parser.add_argument("--symbol", "-s", default="BTCUSDT", help="Trading symbol (for backfill)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Prediction API URL")
    parser.add_argument("--webhook", help="Webhook URL")
    parser.add_argument("--force-retrain", action="store_true", help="Force MLP retraining")
    parser.add_argument("--no-save", action="store_true", help="Don't save signal")
    parser.add_argument("--no-backfill", action="store_true", help="Disable auto-backfill")
    parser.add_argument("--backfill", type=int, metavar="HOURS", help="Manually backfill N hours")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    base_dir = Path(__file__).parent
    
    # Handle init mode
    if args.init:
        if not args.name:
            print("[ERROR] --name is required with --init")
            sys.exit(1)
        init_config(base_dir, args.name, args.init)
        return
    
    # Handle backfill mode
    if args.backfill:
        hist_dir = get_history_dir(base_dir, args.symbol)
        print(f"[BACKFILL] Backfilling {args.backfill} hours for {args.symbol}...")
        api_url = args.api_url or os.environ.get("FINLANG_API_URL", DEFAULT_API_URL)
        count = backfill_missing(hist_dir, args.symbol, args.backfill, api_url)
        print(f"\n[DONE] Backfilled {count} predictions")
        return
    
    # Normal run - requires --name
    if not args.name:
        print("[ERROR] --name is required")
        print("Use --init to create a new config, or --help for more options")
        sys.exit(1)
    
    api_url = args.api_url or os.environ.get("FINLANG_API_URL", DEFAULT_API_URL)
    webhook_url = args.webhook or os.environ.get("FINLANG_WEBHOOK_URL")
    
    signal = run(
        name=args.name,
        api_url=api_url,
        base_dir=base_dir,
        webhook_url=webhook_url,
        force_retrain=args.force_retrain,
        save_signal=not args.no_save,
        auto_backfill=not args.no_backfill,
    )
    
    if args.json:
        print(json.dumps(signal, indent=2, default=str))
    else:
        factors = signal.get("factors", {})
        config = signal.get("config", {})
        print(f"\n{'='*50}")
        print(f"SIGNAL: {signal['name']} ({signal['strategy']})")
        print(f"  Symbol: {signal['symbol']}")
        print(f"  Action: {signal['action']}")
        print(f"  Position: {signal['position']:+d}")
        print(f"  sig_24h: {factors.get('sig_24h', 0):.1f}%")
        
        if "mlp_prob" in factors:
            print(f"  MLP Prob: {factors['mlp_prob']:.1f}%")
        if "bullish_pct" in factors:
            print(f"  Bullish: {factors['bullish_pct']:.1f}% (window={config.get('window')}, thresh={config.get('threshold')})")
        if "volatility" in factors:
            print(f"  Volatility: {factors['volatility']:.2f}%")
        
        print(f"  History: {factors.get('history_count', 0)} records")
        print(f"  Price: ${signal.get('current_close', 0):,.2f}")
        
        if signal.get("rebalance") is True:
            print(f"  Rebalance: YES (new position)")
        elif signal.get("rebalance") is False:
            hours_until = signal.get("hours_until_rebalance", "?")
            print(f"  Rebalance: NO (holding, next in {hours_until}h)")
        
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
