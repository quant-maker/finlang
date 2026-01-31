"""
Finlang - Trading Signal Generation via HF Space API

Lightweight framework for generating trading signals using Kronos predictions
from HF Space. All predictions are fetched via API.

Usage:
    # CLI (recommended)
    python -m finlang.cron --symbol BTCUSDT --strategy voting
    python -m finlang.cron --symbol BTCUSDT --strategy mlp
    python -m finlang.cron --symbol BTCUSDT --strategy dynamic
    
    # Backfill history for cold start
    python -m finlang.cron --symbol BTCUSDT --backfill 300
    
    # Python API
    from finlang.api import get_latest_prediction, get_prediction_for_time
    
    pred = get_latest_prediction()
    print(f"sig_24h: {pred['sig_24h']:.1%}")
"""

from .base import Signal

__version__ = "2.0.0"
__all__ = ["Signal"]
