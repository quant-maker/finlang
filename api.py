"""
Finlang API Client - Fetch predictions from remote prediction service.

Lightweight HTTP-based client that uses raw HTTP requests to call Gradio API endpoints.

Features:
- Fetch real-time predictions (remote service fetches data)
- Send custom OHLCV data for prediction (user-provided data)
- Fetch historical prices from Binance for backfilling

Configuration:
- Set FINLANG_API_URL environment variable to specify the prediction service URL
- Default: reads from environment or uses placeholder
"""

import os
import json
import time
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
import pandas as pd


# Default API configuration (read from environment)
DEFAULT_API_URL = os.environ.get("FINLANG_API_URL", "")
DEFAULT_TIMEOUT = 120
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"


# =============================================================================
# Low-level Gradio API
# =============================================================================

def _call_gradio_api(
    endpoint: str,
    data: list,
    api_url: str = DEFAULT_API_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> Any:
    """
    Call Gradio API using raw HTTP requests.
    
    Gradio uses a 2-step process:
    1. POST to /gradio_api/call/{endpoint} - returns event_id
    2. GET from /gradio_api/call/{endpoint}/{event_id} - returns SSE stream
    """
    if not api_url:
        raise ValueError(
            "API URL not configured. Set FINLANG_API_URL environment variable.\n"
            "Example: export FINLANG_API_URL='https://your-prediction-service.example.com'"
        )
    
    # Step 1: Submit request and get event_id
    submit_url = f"{api_url}/gradio_api/call/{endpoint}"
    resp = requests.post(
        submit_url,
        json={"data": data},
        timeout=30,
    )
    resp.raise_for_status()
    event_id = resp.json()["event_id"]
    
    # Step 2: Poll for result (SSE stream)
    result_url = f"{api_url}/gradio_api/call/{endpoint}/{event_id}"
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        result_resp = requests.get(result_url, timeout=timeout, stream=True)
        result_resp.raise_for_status()
        
        event_type: str = ""
        for line in result_resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                
                if event_type == "complete":
                    result = json.loads(data_str)
                    if isinstance(result, list) and len(result) > 0:
                        return result[0]
                    return result
                elif event_type == "error":
                    raise RuntimeError(f"Gradio API error: {data_str}")
        
        time.sleep(0.5)
    
    raise TimeoutError(f"Gradio API timeout after {timeout}s")


# =============================================================================
# Binance Data Fetching
# =============================================================================

def fetch_binance_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    end_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance public API.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: K-line interval (e.g., "1h", "4h", "1d")
        limit: Number of candles to fetch (max 1000)
        end_time: End time for data (None = now)
    
    Returns:
        DataFrame with columns: timestamps, open, high, low, close, volume, amount
    """
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(limit, 1000),
    }
    
    if end_time:
        params["endTime"] = int(end_time.timestamp() * 1000)
    
    # Try multiple endpoints
    endpoints = [
        BINANCE_API_URL,
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
    ]
    
    last_error = None
    for endpoint in endpoints:
        try:
            resp = requests.get(endpoint, params=params, timeout=30)
            resp.raise_for_status()
            ohlcv = resp.json()
            break
        except Exception as e:
            last_error = e
            continue
    else:
        raise Exception(f"Failed to fetch from all Binance endpoints: {last_error}")
    
    # Parse Binance format
    df = pd.DataFrame(ohlcv, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamps'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['amount'] = pd.to_numeric(df['quote_asset_volume'])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    return df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]


def fetch_binance_ohlcv_range(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a time range, handling pagination.
    
    Args:
        symbol: Trading pair
        interval: K-line interval
        start_time: Start time (None = limit bars before end_time)
        end_time: End time (None = now)
        limit: Total number of bars to fetch
    
    Returns:
        DataFrame with OHLCV data
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    
    all_data = []
    remaining = limit
    current_end = end_time
    
    while remaining > 0:
        batch_size = min(remaining, 1000)
        df = fetch_binance_ohlcv(
            symbol=symbol,
            interval=interval,
            limit=batch_size,
            end_time=current_end,
        )
        
        if df.empty:
            break
        
        all_data.append(df)
        remaining -= len(df)
        
        # Move end time to before the earliest fetched bar
        current_end = df['timestamps'].min() - timedelta(hours=1)
        
        if start_time and current_end < start_time:
            break
    
    if not all_data:
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values('timestamps').reset_index(drop=True)
    result = result.drop_duplicates(subset=['timestamps'], keep='first')
    
    if start_time:
        result = result[result['timestamps'] >= start_time]
    
    return result


def ohlcv_to_json(df: pd.DataFrame) -> str:
    """
    Convert OHLCV DataFrame to JSON format for API.
    
    Args:
        df: DataFrame with timestamps, open, high, low, close, volume, amount
    
    Returns:
        JSON string for predict_custom API
    """
    data = {
        "timestamps": [t.isoformat() for t in df['timestamps']],
        "open": df['open'].tolist(),
        "high": df['high'].tolist(),
        "low": df['low'].tolist(),
        "close": df['close'].tolist(),
        "volume": df['volume'].tolist(),
        "amount": df['amount'].tolist() if 'amount' in df.columns else [0.0] * len(df),
    }
    return json.dumps(data)


# =============================================================================
# Prediction Service API - Prediction Functions
# =============================================================================

def fetch_prediction_simple(
    align_to_hour: bool = True,
    api_url: str = DEFAULT_API_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Fetch simple prediction (remote service fetches current data).
    
    Uses /predict_api endpoint - returns summary stats only.
    """
    result = _call_gradio_api(
        endpoint="predict_api",
        data=[align_to_hour],
        api_url=api_url,
        timeout=timeout,
    )
    return dict(result) if isinstance(result, dict) else {"data": result}


def fetch_prediction_full(
    align_to_hour: bool = True,
    api_url: str = DEFAULT_API_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Fetch full prediction with all Monte Carlo samples.
    
    Uses /predict_all endpoint.
    """
    result = _call_gradio_api(
        endpoint="predict_all",
        data=[align_to_hour],
        api_url=api_url,
        timeout=timeout,
    )
    return dict(result) if isinstance(result, dict) else {"data": result}


def fetch_prediction_custom(
    ohlcv_df: pd.DataFrame,
    pred_horizon: int = 24,
    sample_count: int = 30,
    temperature: float = 1.0,
    top_p: float = 0.95,
    api_url: str = DEFAULT_API_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Send custom OHLCV data to prediction service.
    
    This is the key function for cold start and backfilling.
    
    Args:
        ohlcv_df: DataFrame with timestamps, open, high, low, close, volume
        pred_horizon: Prediction horizon in hours (1-48)
        sample_count: Number of Monte Carlo samples (1-100)
        temperature: Sampling temperature (0.1-2.0)
        top_p: Nucleus sampling probability (0.1-1.0)
        api_url: Prediction service URL
        timeout: Request timeout
    
    Returns:
        Prediction result dict
    """
    # Convert DataFrame to JSON
    ohlcv_json = ohlcv_to_json(ohlcv_df)
    
    # Call predict_custom endpoint
    result = _call_gradio_api(
        endpoint="predict_custom",
        data=[ohlcv_json, pred_horizon, sample_count, temperature, top_p],
        api_url=api_url,
        timeout=timeout,
    )
    
    # Parse result (it's returned as JSON string)
    if isinstance(result, str):
        return json.loads(result)
    return result


# =============================================================================
# High-level Functions
# =============================================================================

def api_result_to_prediction(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert API result to prediction format compatible with strategies.
    """
    current_close = result.get("last_close", 0)
    
    # Get upside probability (the main signal)
    upside = result.get("upside_probability", 50.0)
    sig_24h = upside / 100.0 if upside > 1 else upside
    
    prediction = {
        "timestamp": result.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "current_close": current_close,
        "sig_24h": sig_24h,  # This is the main probability signal
        "upside_probability": sig_24h,
        "source": "prediction_api",
        "symbol": result.get("symbol", "BTCUSDT"),
        "inference_time": result.get("inference_time_seconds", 0),
    }
    
    # Add hourly price predictions if available (NOT overwriting sig_24h)
    if "predictions" in result:
        preds = result["predictions"]
        means = preds.get("mean", [])
        
        for h, pred_mean in enumerate(means, 1):
            if h <= 24:
                # Store predicted price, not binary signal
                prediction[f"pred_{h}h"] = pred_mean
    
    return prediction


def get_latest_prediction(
    align_to_hour: bool = True,
    api_url: str = DEFAULT_API_URL,
    timeout: int = DEFAULT_TIMEOUT,
    simple: bool = True,
) -> Dict[str, Any]:
    """
    Fetch and convert prediction in one call (remote service fetches current data).
    """
    if simple:
        result = fetch_prediction_simple(align_to_hour, api_url, timeout)
    else:
        result = fetch_prediction_full(align_to_hour, api_url, timeout)
    
    return api_result_to_prediction(result)


def get_prediction_for_time(
    target_time: datetime,
    symbol: str = "BTCUSDT",
    hist_bars: int = 384,
    api_url: str = DEFAULT_API_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Get prediction for a specific historical time.
    
    Fetches OHLCV data ending at target_time and sends to API.
    This is used for cold start backfilling.
    
    Args:
        target_time: The time to predict for
        symbol: Trading symbol
        hist_bars: Number of historical bars to include
        api_url: Prediction service URL
        timeout: Request timeout
    
    Returns:
        Prediction dict with sig_24h, current_close, etc.
    """
    # Fetch historical data ending at target_time
    ohlcv_df = fetch_binance_ohlcv(
        symbol=symbol,
        interval="1h",
        limit=hist_bars,
        end_time=target_time,
    )
    
    if ohlcv_df.empty:
        raise ValueError(f"No data available for {target_time}")
    
    # Call custom prediction API
    result = fetch_prediction_custom(
        ohlcv_df=ohlcv_df,
        pred_horizon=24,
        sample_count=30,
        api_url=api_url,
        timeout=timeout,
    )
    
    # Convert to prediction format
    pred = api_result_to_prediction(result)
    pred["data_timestamp"] = target_time.isoformat()
    pred["current_close"] = float(ohlcv_df['close'].iloc[-1])
    
    return pred


def backfill_predictions(
    start_time: datetime,
    end_time: Optional[datetime] = None,
    symbol: str = "BTCUSDT",
    interval_hours: int = 1,
    hist_bars: int = 384,
    api_url: str = DEFAULT_API_URL,
    timeout: int = DEFAULT_TIMEOUT,
    progress_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """
    Backfill predictions for a time range using custom data API.
    
    For each hour in the range, fetches historical data and gets prediction.
    
    Args:
        start_time: Start of backfill range
        end_time: End of backfill range (None = now)
        symbol: Trading symbol
        interval_hours: Hours between predictions
        hist_bars: Historical bars per prediction
        api_url: Prediction service URL
        timeout: Timeout per prediction
        progress_callback: Called with (current, total) for progress updates
    
    Returns:
        List of prediction dicts
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    
    predictions = []
    current = start_time.replace(minute=0, second=0, microsecond=0)
    
    total = int((end_time - current).total_seconds() / 3600 / interval_hours)
    count = 0
    
    while current <= end_time:
        try:
            pred = get_prediction_for_time(
                target_time=current,
                symbol=symbol,
                hist_bars=hist_bars,
                api_url=api_url,
                timeout=timeout,
            )
            predictions.append(pred)
            
            if progress_callback:
                count += 1
                progress_callback(count, total)
                
        except Exception as e:
            print(f"[WARN] Failed to get prediction for {current}: {e}")
        
        current += timedelta(hours=interval_hours)
        
        # Rate limiting to avoid overwhelming the API
        time.sleep(1)
    
    return predictions


def get_trading_signal(
    threshold: float = 0.55,
    api_url: str = DEFAULT_API_URL,
) -> Dict[str, Any]:
    """
    Get a simple trading signal: LONG, FLAT, or SHORT.
    """
    pred = get_latest_prediction(api_url=api_url, simple=True)
    
    sig_24h = pred["sig_24h"]
    
    if sig_24h >= threshold:
        signal = "LONG"
    elif sig_24h <= (1 - threshold):
        signal = "SHORT"
    else:
        signal = "FLAT"
    
    return {
        "signal": signal,
        "sig_24h": sig_24h,
        "current_close": pred["current_close"],
        "timestamp": pred["timestamp"],
        "threshold": threshold,
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Finlang Prediction API Client Test")
    print("=" * 60)
    
    if not DEFAULT_API_URL:
        print("\nError: FINLANG_API_URL environment variable not set")
        print("Please set it to your prediction service URL")
        sys.exit(1)
    
    try:
        # Test 1: Simple API
        print("\n[1] Testing simple API (/predict_api)...")
        pred = get_latest_prediction(simple=True)
        print(f"    Current price: ${pred['current_close']:,.2f}")
        print(f"    24h upside probability: {pred['sig_24h']:.1%}")
        
        # Test 2: Binance data fetch
        print("\n[2] Testing Binance OHLCV fetch...")
        ohlcv = fetch_binance_ohlcv(limit=10)
        print(f"    Fetched {len(ohlcv)} bars")
        print(f"    Latest: {ohlcv['timestamps'].iloc[-1]} @ ${ohlcv['close'].iloc[-1]:,.2f}")
        
        # Test 3: Custom prediction
        print("\n[3] Testing custom prediction API...")
        ohlcv_384 = fetch_binance_ohlcv(limit=384)
        custom_result = fetch_prediction_custom(ohlcv_384)
        custom_pred = api_result_to_prediction(custom_result)
        print(f"    Custom prediction sig_24h: {custom_pred['sig_24h']:.1%}")
        
        # Test 4: Historical prediction
        print("\n[4] Testing historical prediction...")
        past_time = datetime.now(timezone.utc) - timedelta(hours=24)
        hist_pred = get_prediction_for_time(past_time)
        print(f"    Prediction for {past_time.strftime('%Y-%m-%d %H:00')}")
        print(f"    sig_24h: {hist_pred['sig_24h']:.1%}")
        print(f"    price: ${hist_pred['current_close']:,.2f}")
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
