"""
Portfolio Optimizer — Markowitz Mean-Variance Optimization
===========================================================
Three methods:
  A) Standard Markowitz (raw correlation matrix)
  B) RMT-Denoised Markowitz (filtered correlation matrix)
  C) Equal-Weight benchmark (1/N)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import RISK_FREE_RATE


# =============================================================================
# Portfolio Performance Metrics
# =============================================================================

def portfolio_performance(weights, mu, cov, rf=RISK_FREE_RATE):
    """
    Compute portfolio return, volatility, and Sharpe ratio.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N,).
    mu : np.ndarray
        Annualized expected returns (N,).
    cov : np.ndarray
        Annualized covariance matrix (N x N).
    rf : float
        Annualized risk-free rate.

    Returns
    -------
    exp_return, volatility, sharpe : float
    """
    ret = np.dot(weights, mu)
    vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    sharpe = (ret - rf) / vol if vol > 1e-10 else 0.0
    return ret, vol, sharpe


# =============================================================================
# Markowitz Optimization (Maximum Sharpe Ratio)
# =============================================================================

def optimize_markowitz(mu, cov, rf=RISK_FREE_RATE, allow_short=False):
    """
    Find the maximum Sharpe ratio portfolio using scipy.optimize.

    Parameters
    ----------
    mu : np.ndarray
        Annualized expected returns.
    cov : np.ndarray
        Annualized covariance matrix.
    rf : float
        Risk-free rate.
    allow_short : bool
        If False, weights are constrained to [0, 1].

    Returns
    -------
    result : dict
        {'weights', 'return', 'volatility', 'sharpe'}
    """
    n = len(mu)

    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        return -(ret - rf) / vol if vol > 1e-10 else 0.0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if allow_short:
        bounds = [(-1, 1)] * n
    else:
        bounds = [(0, 1)] * n

    x0 = np.ones(n) / n  # Start from equal weight

    result = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    weights = result.x
    # Clip tiny negative weights (numerical artifacts)
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()  # Re-normalize

    exp_ret, vol, sharpe = portfolio_performance(weights, mu, cov, rf)

    return {
        "weights": weights,
        "return": exp_ret,
        "volatility": vol,
        "sharpe": sharpe,
    }


# =============================================================================
# Equal-Weight Portfolio
# =============================================================================

def equal_weight_portfolio(mu, cov, rf=RISK_FREE_RATE):
    """
    1/N equal-weight portfolio.

    Returns
    -------
    result : dict
        {'weights', 'return', 'volatility', 'sharpe'}
    """
    n = len(mu)
    weights = np.ones(n) / n
    exp_ret, vol, sharpe = portfolio_performance(weights, mu, cov, rf)

    return {
        "weights": weights,
        "return": exp_ret,
        "volatility": vol,
        "sharpe": sharpe,
    }


# =============================================================================
# Efficient Frontier
# =============================================================================

def compute_efficient_frontier(mu, cov, rf=RISK_FREE_RATE, n_points=100):
    """
    Compute the efficient frontier by targeting n_points return levels.

    Returns
    -------
    returns, volatilities, sharpes : np.ndarray
        Each of shape (n_points,).
    """
    n = len(mu)

    # Target returns: from min to max achievable
    min_ret = mu.min()
    max_ret = mu.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_returns = []
    frontier_vols = []
    frontier_sharpes = []

    for target in target_returns:
        def portfolio_var(w):
            return np.dot(w, np.dot(cov, w))

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mu) - t},
        ]
        bounds = [(0, 1)] * n
        x0 = np.ones(n) / n

        result = minimize(
            portfolio_var,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )

        if result.success:
            w = result.x
            w = np.clip(w, 0, None)
            if w.sum() > 1e-10:
                w /= w.sum()
                r, v, s = portfolio_performance(w, mu, cov, rf)
                frontier_returns.append(r)
                frontier_vols.append(v)
                frontier_sharpes.append(s)

    return np.array(frontier_returns), np.array(frontier_vols), np.array(frontier_sharpes)


# =============================================================================
# Multiple Method Optimization
# =============================================================================

def optimize_all_methods(train_returns, rf=RISK_FREE_RATE):
    """
    Run all three optimization methods on training data.

    Parameters
    ----------
    train_returns : pd.DataFrame
        Training period daily log-returns.

    Returns
    -------
    results : dict
        Keys: 'raw_markowitz', 'denoised_markowitz', 'equal_weight'
        Each contains: weights, return, volatility, sharpe, mu, cov_raw, cov_denoised
    """
    from src.rmt_analysis import build_correlation_matrix, denoise_correlation_matrix

    # Annualize: 252 trading days
    mu = train_returns.mean().values * 252
    corr, N, T = build_correlation_matrix(train_returns)

    # Raw covariance
    daily_std = train_returns.std().values
    cov_raw_daily = corr.values * np.outer(daily_std, daily_std)
    cov_raw = cov_raw_daily * 252

    # Denoised covariance
    corr_denoised, _, _ = denoise_correlation_matrix(corr, T)
    cov_denoised_daily = corr_denoised.values * np.outer(daily_std, daily_std)
    cov_denoised = cov_denoised_daily * 252

    # Method A: Raw Markowitz
    raw = optimize_markowitz(mu, cov_raw, rf)
    raw["mu"] = mu
    raw["cov"] = cov_raw

    # Method B: Denoised Markowitz
    denoised = optimize_markowitz(mu, cov_denoised, rf)
    denoised["mu"] = mu
    denoised["cov"] = cov_denoised

    # Method C: Equal Weight
    ew = equal_weight_portfolio(mu, cov_raw, rf)
    ew["mu"] = mu
    ew["cov"] = cov_raw

    return {
        "raw_markowitz": raw,
        "denoised_markowitz": denoised,
        "equal_weight": ew,
    }


# =============================================================================
# Risk Contribution Analysis
# =============================================================================

def risk_contribution(weights, cov, tickers=None):
    """
    Compute per-asset risk contribution to total portfolio volatility.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N,).
    cov : np.ndarray
        Annualized covariance matrix (N x N).
    tickers : list[str], optional
        Stock names for the index.

    Returns
    -------
    pd.DataFrame
        Columns: Weight, Marginal Risk, Risk Contribution, Pct Contribution (%).
    """
    port_vol = np.sqrt(weights @ cov @ weights)
    marginal_risk = (cov @ weights) / port_vol
    total_rc = weights * marginal_risk
    pct_rc = total_rc / port_vol * 100

    result = pd.DataFrame({
        "Weight": weights,
        "Marginal Risk": marginal_risk,
        "Risk Contribution": total_rc,
        "Pct Contribution (%)": pct_rc,
    }, index=tickers if tickers is not None else range(len(weights)))

    return result


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    from data_fetcher import fetch_stock_data, compute_log_returns, train_test_split

    prices = fetch_stock_data()
    log_ret = compute_log_returns(prices)
    train, test = train_test_split(log_ret)

    results = optimize_all_methods(train)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Return: {res['return']:.2%}")
        print(f"  Vol:    {res['volatility']:.2%}")
        print(f"  Sharpe: {res['sharpe']:.3f}")
