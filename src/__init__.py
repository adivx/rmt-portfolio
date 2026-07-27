"""
RMT Portfolio Optimization — Source Package
============================================
Random Matrix Theory applied to Indian Stock Market Portfolio Optimization.
"""

from src.data_fetcher import fetch_stock_data, compute_log_returns, train_test_split
from src.rmt_analysis import (
    build_correlation_matrix,
    marchenko_pastur_pdf,
    marchenko_pastur_bounds,
    eigenvalue_analysis_with_T,
    denoise_correlation_matrix,
    participation_ratio,
)
from src.portfolio_optimizer import (
    optimize_markowitz,
    equal_weight_portfolio,
    compute_efficient_frontier,
    optimize_all_methods,
    risk_contribution,
)
from src.backtester import run_backtest, compute_metrics, compute_max_drawdown

__all__ = [
    "fetch_stock_data", "compute_log_returns", "train_test_split",
    "build_correlation_matrix", "eigenvalue_analysis_with_T",
    "denoise_correlation_matrix", "marchenko_pastur_pdf", "marchenko_pastur_bounds",
    "participation_ratio",
    "optimize_markowitz", "equal_weight_portfolio",
    "compute_efficient_frontier", "optimize_all_methods", "risk_contribution",
    "run_backtest", "compute_metrics", "compute_max_drawdown",
]
