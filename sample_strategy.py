"""
MIG Quant Competition — Sample Strategy
========================================
Your submission must define a top-level function:
    get_actions(prices: np.ndarray) -> np.ndarray
Arguments
---------
prices : np.ndarray, shape (num_stocks, num_days)
    Open price for each stock on each trading day.
    Rows are stocks sorted alphabetically by ticker.
    Columns are days in chronological order.
Returns
-------
actions : np.ndarray, shape (num_stocks, num_days)
    Number of shares to trade per stock per day.
      +N  →  buy N shares
      -N  →  sell / open short N shares
       0  →  hold (no trade)
    Values are rounded to the nearest integer (no fractional shares).
Competition Rules (summary)
---------------------------
- Starting capital: $25,000
- Fractional shares: NOT supported
- Runtime limit: 60 seconds
- Memory limit: 512 MB
- No network access inside the sandbox
- No file I/O inside the sandbox
- Strategy must be deterministic
Available packages (pre-installed in sandbox)
---------------------------------------------
numpy>=1.26, pandas>=2.0, scipy, scikit-learn>=1.3,
statsmodels, ta-lib>=0.6.5, numba, joblib
For extra packages, include a requirements.txt inside a .zip submission.
"""

import numpy as np

SHORT_WIN      = 5
LONG_WIN       = 20
TARGET_DOLLARS = 800

def _rolling_mean(matrix: np.ndarray, window: int) -> np.ndarray:
    # To get the rolling sum, subtract the sum from 'window' days ago from the running total
    cs = np.cumsum(matrix, axis=1)
    cs[:, window:] = cs[:, window:] - cs[:, :-window]
    out = cs / window
    # The first (window-1) columns don't have enough history to be useful
    out[:, :window - 1] = np.nan
    return out

def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    fast    = _rolling_mean(prices, SHORT_WIN)
    slow    = _rolling_mean(prices, LONG_WIN)
    # 1 where we want to be long, 0 where we want to be flat
    desired = np.where(fast > slow, 1.0, 0.0)
    # No signal until we have enough history to calculate the slow MA
    desired[:, :LONG_WIN] = 0.0
    # Only trade on days where the desired position changes
    signals = np.diff(desired, prepend=0.0, axis=1)
    # Size each trade so it costs roughly TARGET_DOLLARS regardless of share price
    shares = np.floor(TARGET_DOLLARS / np.where(prices < 1e-9, 1.0, prices))
    shares = np.clip(shares, 1, 100)
    sized  = np.round(shares * np.sign(signals))
    return sized.astype(np.float64)
