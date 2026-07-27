"""
Backtester — Out-of-Sample Portfolio Backtesting
==================================================
Expanding window with monthly rebalancing.
Compares Method A (Raw Markowitz), Method B (RMT-Denoised), Method C (Equal Weight).
"""

import numpy as np
import pandas as pd
from config import RISK_FREE_RATE


def compute_max_drawdown(cumulative_returns):
    """Compute maximum drawdown from a cumulative return series."""
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1.0
    return drawdown


def compute_metrics(daily_returns, rf=RISK_FREE_RATE, label=""):
    """
    Compute portfolio performance metrics.

    Returns
    -------
    dict with keys: annual_return, annual_vol, sharpe, max_drawdown, calmar, win_rate
    """
    daily_rf = rf / 252

    # Cumulative returns
    cum = (1 + daily_returns).cumprod()
    total_days = len(daily_returns)
    years = total_days / 252

    # Annualized return (CAGR)
    total_return = cum.iloc[-1] / cum.iloc[0]
    annual_return = total_return ** (1 / years) - 1

    # Annualized volatility
    annual_vol = daily_returns.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe = (annual_return - rf) / annual_vol if annual_vol > 1e-10 else 0.0

    # Maximum drawdown
    drawdown = compute_max_drawdown(cum)
    max_dd = drawdown.min()

    # Calmar ratio
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # Win rate (fraction of positive months)
    monthly = daily_returns.resample("M").sum()
    win_rate = (monthly > 0).mean()

    metrics = {
        "Annual Return": f"{annual_return:.2%}",
        "Annual Volatility": f"{annual_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Calmar Ratio": f"{calmar:.3f}",
        "Win Rate (Monthly)": f"{win_rate:.1%}",
    }

    if label:
        print(f"\n{'='*40}")
        print(f"  {label}")
        print(f"{'='*40}")
        for k, v in metrics.items():
            print(f"  {k:25s} {v}")

    return metrics, cum, drawdown


def run_backtest(
    log_returns,
    rebalance_freq="M",
    train_years=3,
    rf=RISK_FREE_RATE,
    transaction_cost_bps=0,
):
    """
    Run out-of-sample backtest with expanding window and monthly rebalancing.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Full period daily log-returns.
    rebalance_freq : str
        Pandas resample frequency ('M' = monthly).
    train_years : int
        Initial training window in years.
    transaction_cost_bps : float
        Transaction cost in basis points (0 = no costs).

    Returns
    -------
    results : dict
        Keys: 'method_a', 'method_b', 'method_c'
        Each: dict with 'daily_returns', 'cumulative', 'drawdown', 'weights_history'
    metrics_df : pd.DataFrame
        Comparison table of all methods.
    """
    from src.rmt_analysis import build_correlation_matrix, denoise_correlation_matrix
    from src.portfolio_optimizer import optimize_markowitz, equal_weight_portfolio

    start_date = log_returns.index[0]
    train_end = start_date + pd.DateOffset(years=train_years)
    # Find nearest trading day
    train_end = log_returns.index[log_returns.index <= train_end][-1]

    # Get rebalance dates (month starts after training period)
    test_data = log_returns.loc[train_end:]
    rebalance_dates = test_data.resample(rebalance_freq).apply(lambda x: x.index[0])
    rebalance_dates = rebalance_dates[rebalance_dates >= train_end]

    # Storage
    method_a_daily = []
    method_b_daily = []
    method_c_daily = []
    weights_history = {"A": [], "B": [], "C": []}

    current_weights = {"A": None, "B": None, "C": None}

    print(f"[BT] Training period: {start_date.date()} to {train_end.date()}")
    print(f"[BT] Test period: {train_end.date()} to {test_data.index[-1].date()}")
    print(f"[BT] Rebalancing {len(rebalance_dates)} times...")

    for i, rebal_date in enumerate(rebalance_dates):
        # Use all data up to (but not including) rebalance date
        history = log_returns.loc[:rebal_date - pd.Timedelta(days=1)]

        if len(history) < 60:  # Need at least ~3 months of data
            continue

        # Determine next rebalance date (or end of data)
        if i + 1 < len(rebalance_dates):
            next_rebal = rebalance_dates.iloc[i + 1]
        else:
            next_rebal = test_data.index[-1] + pd.Timedelta(days=1)

        # Get returns for this period
        period_returns = test_data.loc[rebal_date:next_rebal - pd.Timedelta(days=1)]
        if len(period_returns) == 0:
            continue

        # Build correlation matrix
        corr, N, T = build_correlation_matrix(history)
        daily_std = history.std().values

        # --- Method A: Raw Markowitz ---
        cov_raw_daily = corr.values * np.outer(daily_std, daily_std)
        cov_raw = cov_raw_daily * 252
        mu = history.mean().values * 252

        try:
            res_a = optimize_markowitz(mu, cov_raw, rf)
            weights_a = res_a["weights"]
        except Exception:
            weights_a = np.ones(N) / N

        # --- Method B: Denoised Markowitz ---
        corr_denoised, _, _ = denoise_correlation_matrix(corr, T)
        cov_denoised_daily = corr_denoised.values * np.outer(daily_std, daily_std)
        cov_denoised = cov_denoised_daily * 252

        try:
            res_b = optimize_markowitz(mu, cov_denoised, rf)
            weights_b = res_b["weights"]
        except Exception:
            weights_b = np.ones(N) / N

        # --- Method C: Equal Weight ---
        weights_c = np.ones(N) / N

        # Compute daily portfolio returns for this period
        period_ret_matrix = period_returns.values  # (days x N)
        ret_a = period_ret_matrix @ weights_a
        ret_b = period_ret_matrix @ weights_b
        ret_c = period_ret_matrix @ weights_c

        # Transaction cost: one-time drag on first day of period
        if transaction_cost_bps > 0:
            tc_rate = transaction_cost_bps / 10000
            if weights_history["A"]:
                turnover_a = np.sum(np.abs(weights_a - weights_history["A"][-1]))
                turnover_b = np.sum(np.abs(weights_b - weights_history["B"][-1]))
                ret_a = ret_a.copy()
                ret_b = ret_b.copy()
                ret_a[0] -= turnover_a * tc_rate
                ret_b[0] -= turnover_b * tc_rate
            # Method C (equal weight) has no turnover after initial allocation
            # unless N changes between periods

        method_a_daily.append(pd.Series(ret_a, index=period_returns.index))
        method_b_daily.append(pd.Series(ret_b, index=period_returns.index))
        method_c_daily.append(pd.Series(ret_c, index=period_returns.index))

        weights_history["A"].append(weights_a.copy())
        weights_history["B"].append(weights_b.copy())
        weights_history["C"].append(weights_c.copy())

    # Concatenate all periods
    daily_a = pd.concat(method_a_daily) if method_a_daily else pd.Series(dtype=float)
    daily_b = pd.concat(method_b_daily) if method_b_daily else pd.Series(dtype=float)
    daily_c = pd.concat(method_c_daily) if method_c_daily else pd.Series(dtype=float)

    # Compute metrics
    print("\n" + "=" * 60)
    print("  OUT-OF-SAMPLE BACKTEST RESULTS")
    print("=" * 60)

    metrics_a, cum_a, dd_a = compute_metrics(daily_a, rf, label="Method A: Raw Markowitz")
    metrics_b, cum_b, dd_b = compute_metrics(daily_b, rf, label="Method B: RMT-Denoised Markowitz")
    metrics_c, cum_c, dd_c = compute_metrics(daily_c, rf, label="Method C: Equal Weight (1/N)")

    # Summary DataFrame
    metrics_df = pd.DataFrame([metrics_a, metrics_b, metrics_c],
                               index=["Raw Markowitz", "RMT-Denoised", "Equal Weight"])

    results = {
        "method_a": {"daily": daily_a, "cumulative": cum_a, "drawdown": dd_a, "weights": weights_history["A"]},
        "method_b": {"daily": daily_b, "cumulative": cum_b, "drawdown": dd_b, "weights": weights_history["B"]},
        "method_c": {"daily": daily_c, "cumulative": cum_c, "drawdown": dd_c, "weights": weights_history["C"]},
    }

    return results, metrics_df


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    from data_fetcher import fetch_stock_data, compute_log_returns

    prices = fetch_stock_data()
    log_ret = compute_log_returns(prices)
    results, metrics = run_backtest(log_ret)
    print("\n", metrics)
