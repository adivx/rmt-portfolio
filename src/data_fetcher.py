"""
Data Fetcher — NSE Stock Data Download & Preprocessing
======================================================
Downloads NIFTY 50 stock data from Yahoo Finance, handles failures gracefully,
computes log-returns, and provides train/test splits.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from config import (
    NIFTY50_TICKERS, START_DATE, END_DATE,
    DATA_CACHE_DIR, MIN_STOCKS
)


def fetch_stock_data(
    tickers=None,
    start=START_DATE,
    end=END_DATE,
    cache_dir=DATA_CACHE_DIR,
    use_cache=True,
):
    """
    Download NIFTY 50 daily closing prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str], optional
        Stock tickers with .NS suffix. Defaults to NIFTY50_TICKERS.
    start, end : str
        Date range in 'YYYY-MM-DD' format.
    cache_dir : str
        Directory to cache downloaded CSV files.
    use_cache : bool
        If True, load from cache if available.

    Returns
    -------
    prices : pd.DataFrame
        Daily adjusted closing prices, one column per ticker.
    """
    if tickers is None:
        tickers = NIFTY50_TICKERS

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "nifty50_prices.csv")

    # --- Try cache first ---
    if use_cache and os.path.exists(cache_path):
        print("[DATA] Loading cached data from", cache_path)
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if prices.shape[1] >= MIN_STOCKS:
            print(f"[DATA] Loaded {prices.shape[1]} stocks, {len(prices)} trading days")
            return prices
        print("[DATA] Cache insufficient, re-downloading...")

    # --- Download from Yahoo Finance ---
    print(f"[DATA] Downloading {len(tickers)} NIFTY 50 stocks...")
    print(f"[DATA] Period: {start} to {end}")

    # Try batch download first
    failed_tickers = []
    try:
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=True,
            threads=True,
        )
        # yfinance returns multi-level columns; extract Close
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        elif "Close" in raw.columns:
            # Single stock returned
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            # Fallback: try first numeric column
            prices = raw.iloc[:, :1]
            prices.columns = [tickers[0]]

    except Exception as e:
        print(f"[DATA] Batch download failed ({e}), falling back to individual...")
        prices = pd.DataFrame()
        for ticker in tqdm(tickers, desc="Downloading"):
            try:
                df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                if len(df) > 100:
                    prices[ticker] = df["Close"]
            except Exception:
                failed_tickers.append(ticker)

    # --- Validate & clean ---
    n_initial = prices.shape[1]
    prices = _validate_and_clean(prices, min_stocks=MIN_STOCKS)
    n_final = prices.shape[1]

    if failed_tickers:
        print(f"[DATA] Failed tickers ({len(failed_tickers)}): {failed_tickers}")
    print(f"[DATA] Usable stocks: {n_final}/{n_initial} | Trading days: {len(prices)}")

    # --- Cache ---
    prices.to_csv(cache_path)
    print(f"[DATA] Cached to {cache_path}")

    return prices


def _validate_and_clean(prices, min_stocks=MIN_STOCKS):
    """
    Validate downloaded data: drop stocks with too many missing values,
    forward-fill remaining gaps, require minimum stock count.
    """
    if prices.empty:
        raise ValueError("[DATA] No data downloaded. Check network connection.")

    # Drop stocks with > 5% missing data
    max_missing = int(len(prices) * 0.05)
    missing_counts = prices.isna().sum()
    valid_stocks = missing_counts[missing_counts <= max_missing].index
    prices = prices[valid_stocks]

    # Forward-fill then backward-fill remaining NaN (holidays, suspensions)
    prices = prices.ffill().bfill()

    # Drop any remaining rows with NaN
    prices = prices.dropna()

    if prices.shape[1] < min_stocks:
        raise ValueError(
            f"[DATA] Only {prices.shape[1]} stocks passed validation. "
            f"Need at least {min_stocks}."
        )

    return prices


def compute_log_returns(prices):
    """
    Compute continuously compounded log-returns from price levels.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily closing prices.

    Returns
    -------
    log_returns : pd.DataFrame
        Daily log-returns, first row dropped.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def train_test_split(log_returns, train_years=3, start_date=None):
    """
    Split log-returns into train/test sets.

    Parameters
    ----------
    log_returns : pd.DataFrame
    train_years : int
        Number of years for training window.

    Returns
    -------
    train, test : pd.DataFrame
    """
    if start_date is None:
        start_date = log_returns.index[0]

    train_end = start_date + pd.DateOffset(years=train_years)
    # Find the nearest actual trading day
    train_end = log_returns.index[log_returns.index <= train_end][-1]

    train = log_returns.loc[:train_end]
    test = log_returns.loc[train_end:]

    return train, test


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    prices = fetch_stock_data()
    log_ret = compute_log_returns(prices)
    train, test = train_test_split(log_ret)
    print(f"\nTrain: {train.shape} | Test: {test.shape}")
    print(f"Train period: {train.index[0]} to {train.index[-1]}")
    print(f"Test period:  {test.index[0]} to {test.index[-1]}")
