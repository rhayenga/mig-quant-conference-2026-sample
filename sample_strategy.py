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
import talib

SHORT_WIN          = 5
LONG_WIN           = 20
SHARPE_LOOKBACK    = 60
STARTING_CAPITAL   = 25_000.0
PAIRS_Z_WINDOW     = 20
PAIRS_Z_ENTRY      = 1.5
PAIRS_Z_EXIT       = 0.25
MAX_GROSS_EXPOSURE = 0.95
RSI_PERIOD         = 14
RSI_OVERSOLD       = 30    # only buy when RSI is below this
RSI_OVERBOUGHT     = 70    # only sell when RSI is above this

def _rolling_mean(matrix: np.ndarray, window: int) -> np.ndarray:
    # To get the rolling sum, subtract the sum from 'window' days ago from the running total
    cs = np.cumsum(matrix, axis=1)
    cs[:, window:] = cs[:, window:] - cs[:, :-window]
    out = cs / window
    # The first (window-1) columns don't have enough history to be useful
    out[:, :window - 1] = np.nan
    return out

def _rolling_std(matrix: np.ndarray, window: int) -> np.ndarray:
    mu  = _rolling_mean(matrix, window)
    mu2 = _rolling_mean(matrix ** 2, window)
    var = np.where((mu2 - mu ** 2) < 0, 0.0, mu2 - mu ** 2)
    return np.sqrt(var)

def _sharpe_weights(prices: np.ndarray) -> np.ndarray:
    returns = np.diff(prices, axis=1) / np.where(prices[:, :-1] < 1e-9, 1e-9, prices[:, :-1])
    returns = np.concatenate([np.zeros((prices.shape[0], 1)), returns], axis=1)

    mu     = _rolling_mean(returns, SHARPE_LOOKBACK)
    std    = _rolling_std(returns, SHARPE_LOOKBACK)
    std    = np.where(std < 1e-9, 1e-9, std)

    sharpe = np.clip(np.where(np.isnan(mu / std), 0.0, mu / std), 0.0, None)
    totals = np.where(sharpe.sum(axis=0, keepdims=True) < 1e-9, 1.0,
                      sharpe.sum(axis=0, keepdims=True))
    return sharpe / totals

def _compute_rsi(prices: np.ndarray) -> np.ndarray:
    # Compute RSI for every stock and return a matrix of the same shape as prices
    num_stocks, num_days = prices.shape
    rsi = np.full((num_stocks, num_days), 50.0)  # default to neutral 50 during warm-up
    for i in range(num_stocks):
        rsi[i] = talib.RSI(prices[i].astype(float), timeperiod=RSI_PERIOD)
    # Fill any NaNs from the warm-up period with 50 so they never trigger a signal
    rsi = np.where(np.isnan(rsi), 50.0, rsi)
    return rsi

def _find_best_pair(prices: np.ndarray):
    # Compute the correlation matrix across all stocks and pick the most correlated pair
    returns = np.diff(prices, axis=1) / np.where(prices[:, :-1] < 1e-9, 1e-9, prices[:, :-1])
    corr = np.corrcoef(returns)
    # Mask the diagonal so we don't pick a stock paired with itself
    np.fill_diagonal(corr, -1.0)
    idx = np.unravel_index(np.argmax(corr), corr.shape)
    return int(idx[0]), int(idx[1])

def _pairs_overlay(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    pa = np.zeros((num_stocks, num_days))

    # Detect the most correlated pair from the data — no hardcoded indices
    idx_a, idx_b = _find_best_pair(prices)

    ratio  = prices[idx_a] / np.where(prices[idx_b] < 1e-9, 1e-9, prices[idx_b])
    mu_r   = _rolling_mean(ratio[np.newaxis, :], PAIRS_Z_WINDOW)[0]
    std_r  = _rolling_std(ratio[np.newaxis, :],  PAIRS_Z_WINDOW)[0]
    std_r  = np.where(std_r < 1e-9, 1e-9, std_r)
    zscore = (ratio - mu_r) / std_r

    pos   = 0
    alloc = 0.10 * STARTING_CAPITAL

    for t in range(PAIRS_Z_WINDOW, num_days):
        z  = zscore[t]
        sa = int(alloc / (2 * prices[idx_a, t]))
        sb = int(alloc / (2 * prices[idx_b, t]))

        if pos == 0:
            if z > PAIRS_Z_ENTRY and sa > 0 and sb > 0:
                # Stock A expensive relative to B — short A, long B
                pa[idx_a, t] -= sa
                pa[idx_b, t] += sb
                pos = -1
            elif z < -PAIRS_Z_ENTRY and sa > 0 and sb > 0:
                # Stock B expensive relative to A — long A, short B
                pa[idx_a, t] += sa
                pa[idx_b, t] -= sb
                pos = +1
        else:
            if abs(z) < PAIRS_Z_EXIT:
                # Spread has collapsed — close both legs
                pa[idx_a, t] += sa * (-pos)
                pa[idx_b, t] += sb * ( pos)
                pos = 0

    return pa

def _guardrail(actions: np.ndarray, prices: np.ndarray) -> np.ndarray:
    safe      = actions.copy()
    cash      = STARTING_CAPITAL
    positions = np.zeros(prices.shape[0])

    for t in range(prices.shape[1]):
        # Gross value of all open short positions
        gross_short = (-np.minimum(positions, 0) * prices[:, t]).sum()

        # If shorts are too large relative to cash, scale today's trades down
        if cash > 0:
            exposure_ratio = gross_short / cash
            if exposure_ratio > MAX_GROSS_EXPOSURE:
                safe[:, t] *= MAX_GROSS_EXPOSURE / exposure_ratio

        # Enforce position limit after scaling
        safe[:, t] = np.clip(safe[:, t], -100, 100)

        # Update running cash and positions after today's trades
        cash      -= (safe[:, t] * prices[:, t]).sum()
        positions += safe[:, t]

        # Hard stop — if cash goes negative, zero out all trades today
        if cash < 0:
            cash      += (safe[:, t] * prices[:, t]).sum()
            positions -= safe[:, t]
            safe[:, t] = 0.0

    return np.round(safe).astype(np.float64)

def get_actions(prices: np.ndarray) -> np.ndarray:
    """
    Vectorized MA crossover + RSI filter + Sharpe sizing + pairs trade + debt guardrail.

    RSI acts as a filter on top of the MA crossover signal:
      - A buy signal only executes if RSI is below 30 (stock is oversold)
      - A sell signal only executes if RSI is above 70 (stock is overbought)
    This filters out crossovers that happen in choppy, directionless markets
    and improves the quality of every trade rather than the quantity.

    Capped at 100 shares per stock to satisfy the position limit.
    """
    num_stocks, num_days = prices.shape

    fast = _rolling_mean(prices, SHORT_WIN)
    slow = _rolling_mean(prices, LONG_WIN)

    # 1 where we want to be long, 0 where we want to be flat
    desired = np.where(fast > slow, 1.0, 0.0)
    # No signal until we have enough history to calculate the slow MA
    desired[:, :LONG_WIN] = 0.0
    # Only trade on days where the desired position changes
    signals = np.diff(desired, prepend=0.0, axis=1)

    # RSI filter — only buy when oversold, only sell when overbought
    rsi        = _compute_rsi(prices)
    buy_filter = rsi < RSI_OVERSOLD    # True on days we allow a buy
    sell_filter = rsi > RSI_OVERBOUGHT  # True on days we allow a sell
    signals    = np.where(signals > 0, signals * buy_filter,
                 np.where(signals < 0, signals * sell_filter, 0.0))

    weights    = _sharpe_weights(prices)
    alloc      = STARTING_CAPITAL * weights
    raw_shares = alloc / np.where(prices < 1e-9, 1.0, prices)
    sized      = np.clip(np.round(raw_shares * np.sign(signals)), -100, 100)

    sized += _pairs_overlay(prices)
    sized  = np.clip(sized, -100, 100)

    return _guardrail(sized, prices)
