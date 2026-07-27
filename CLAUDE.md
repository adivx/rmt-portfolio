# RMT Portfolio Optimization

## Project Overview
Random Matrix Theory applied to NIFTY 50 stock portfolio optimization.
Downloads Indian stock data from Yahoo Finance, denoises correlation matrices
using Marchenko-Pastur theory, and benchmarks against standard Markowitz.

## Build & Run
- Install: `pip install -r requirements.txt`
- Generate notebook: `python create_notebook.py`
- Run notebook: `jupyter notebook notebooks/rmt_portfolio_optimization.ipynb`
- Run backtest: `python src/backtester.py`

## Architecture
| File | Role |
|------|------|
| `config.py` | All constants (tickers, dates, colors, sector mapping, params) |
| `src/data_fetcher.py` | Yahoo Finance download + preprocessing |
| `src/rmt_analysis.py` | Core RMT: MP distribution, eigenvalue analysis, denoising |
| `src/portfolio_optimizer.py` | Markowitz optimization (raw, denoised, equal-weight) + risk contribution |
| `src/backtester.py` | Expanding-window out-of-sample backtest with transaction costs |
| `src/visualization.py` | 16+ visualization functions (matplotlib + Plotly) |
| `create_notebook.py` | Generates the .ipynb notebook programmatically |

## Key Conventions
- Risk-free rate: 7% (India 10Y bond yield)
- Time period: 2019-01-01 to 2024-12-31
- Training: first 3 years, Testing: last 2 years
- Monthly rebalancing with expanding window
- All visualizations use dark mode (COLORS dict from config.py)
- Transaction costs in basis points (default 0)

## Data Flow
1. `data_fetcher` → prices DataFrame → `compute_log_returns`
2. `rmt_analysis` → eigenvalues → denoised correlation matrix
3. `portfolio_optimizer` → weights for 3 methods
4. `backtester` → out-of-sample daily returns + metrics
5. `visualization` → charts and dashboard

## Visualization Types
- **Static (matplotlib):** eigenvalue spectrum, heatmaps, efficient frontier, cumulative returns, rolling Sharpe, drawdown, weight allocation, dashboard, eigenvector heatmap
- **Interactive (Plotly):** correlation network, 3D frontier, Sankey, return race, risk contribution, sector correlation
- **Animated (matplotlib):** rolling eigenvalue spectrum over time
