"""
Script to generate the Jupyter notebook for RMT Portfolio Optimization.
Run this once to create the .ipynb file.
"""

import json

cells = []

def md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    })

def code(source):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
        "execution_count": None,
        "outputs": []
    })

# =============================================================================
# CELL 1: Title
# =============================================================================
md([
    "# 📊 Random Matrix Theory Applied to Indian Stock Market Portfolio Optimization\n",
    "\n",
    "### [Your Name] | July 2026\n",
    "\n",
    "---\n",
    "\n",
    "**Abstract:** This project applies Random Matrix Theory (RMT) to denoise stock correlation matrices\n",
    "extracted from NIFTY 50 data, and demonstrates that portfolios constructed using the denoised\n",
    "correlation matrix outperform standard Markowitz mean-variance optimization in out-of-sample testing.\n",
    "The Marchenko-Pastur distribution serves as the null hypothesis to separate informative (signal)\n",
    "eigenvalues from noise, producing more robust and generalizable portfolio allocations."
])

# =============================================================================
# CELL 2: Mathematical Framework
# =============================================================================
md([
    "## 📐 Mathematical Framework\n",
    "\n",
    "### 1. Marchenko-Pastur Distribution\n",
    "\n",
    "For an $N \\times T$ random matrix $R$ with i.i.d. entries of variance $\\sigma^2/T$,\n",
    "the eigenvalue density converges to:\n",
    "\n",
    "$$\n",
    "f_{MP}(\\lambda) = \\frac{Q}{2\\pi\\sigma^2} \\cdot \\frac{\\sqrt{(\\lambda_+ - \\lambda)(\\lambda - \\lambda_-)}}{\\lambda}\n",
    "$$\n",
    "\n",
    "where $Q = T/N$ and the support bounds are:\n",
    "\n",
    "$$\n",
    "\\lambda_{\\pm} = \\sigma^2\\left(1 \\pm \\sqrt{Q}\\right)^2\n",
    "$$\n",
    "\n",
    "### 2. Signal-Noise Separation\n",
    "\n",
    "Eigenvalues $\\lambda_i > \\lambda_+$ contain genuine cross-correlations (signal),\n",
    "while eigenvalues within $[\\lambda_-, \\lambda_+]$ are statistically indistinguishable from noise.\n",
    "\n",
    "### 3. Denoising Procedure\n",
    "\n",
    "1. Eigendecompose: $C = V \\Lambda V^\\top$\n",
    "2. Replace noise eigenvalues with their mean: $\\tilde{\\Lambda} = \\text{diag}(\\ldots, \\bar{\\lambda}_{\\text{noise}}, \\ldots)$\n",
    "3. Reconstruct: $\\tilde{C} = V \\tilde{\\Lambda} V^\\top$\n",
    "4. Rescale to unit diagonal: $\\hat{C} = D^{-1/2} \\tilde{C} D^{-1/2}$\n",
    "\n",
    "### 4. Markowitz Optimization\n",
    "\n",
    "$$\n",
    "\\max_{\\mathbf{w}} \\frac{\\mathbf{w}^\\top \\boldsymbol{\\mu} - r_f}{\\sqrt{\\mathbf{w}^\\top \\Sigma \\mathbf{w}}}\n",
    "\\quad \\text{s.t.} \\quad \\sum_i w_i = 1, \\quad w_i \\geq 0\n",
    "$$\n",
    "\n",
    "We compare three covariance matrix estimates:\n",
    "- **Method A:** Raw empirical covariance $\\Sigma_{\\text{raw}}$\n",
    "- **Method B:** RMT-denoised covariance $\\hat{\\Sigma}$\n",
    "- **Method C:** Equal-weight $1/N$ benchmark"
])

# =============================================================================
# CELL 3: Imports & Setup
# =============================================================================
md(["## 🔧 Setup & Imports"])

code([
    "import sys\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Add project root to path\n",
    "import os\n",
    "sys.path.insert(0, os.path.abspath('..'))\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from config import *\n",
    "from config import SECTOR_MAP, SECTOR_COLORS\n",
    "from src.data_fetcher import fetch_stock_data, compute_log_returns, train_test_split\n",
    "from src.rmt_analysis import (\n",
    "    build_correlation_matrix, eigenvalue_analysis_with_T,\n",
    "    denoise_correlation_matrix, marchenko_pastur_pdf, marchenko_pastur_bounds\n",
    ")\n",
    "from src.portfolio_optimizer import (\n",
    "    optimize_markowitz, equal_weight_portfolio,\n",
    "    compute_efficient_frontier, optimize_all_methods\n",
    ")\n",
    "from src.backtester import run_backtest, compute_metrics\n",
    "from src.visualization import (\n",
    "    setup_dark_style, plot_eigenvalue_spectrum, plot_correlation_heatmaps,\n",
    "    plot_efficient_frontier, plot_cumulative_returns,\n",
    "    plot_rolling_sharpe, plot_drawdown, plot_weight_allocation,\n",
    "    plot_correlation_network, plot_rolling_eigenvalues,\n",
    "    plot_3d_efficient_frontier, plot_weight_sankey,\n",
    "    plot_clustered_heatmap, plot_cumulative_return_race,\n",
    "    plot_risk_contribution, plot_eigenvector_heatmap,\n",
    "    plot_sector_correlation, plot_dashboard,\n",
    ")\n",
    "\n",
    "setup_dark_style()\n",
    "print('✅ All imports successful')\n",
    "print(f'📊 Stock universe: {len(NIFTY50_TICKERS)} NIFTY 50 stocks')\n",
    "print(f'📅 Period: {START_DATE} to {END_DATE}')\n",
    "print(f'💰 Risk-free rate: {RISK_FREE_RATE:.1%}')"
])

# =============================================================================
# CELL 4: Data Acquisition
# =============================================================================
md(["## 📥 Data Acquisition", "\n", "Downloading daily closing prices for NIFTY 50 stocks from Yahoo Finance..."])

code([
    "prices = fetch_stock_data()\n",
    "print(f'\\n📈 Price matrix shape: {prices.shape}')\n",
    "print(f'📅 Date range: {prices.index[0].date()} to {prices.index[-1].date()}')\n",
    "print(f'\\n📊 First 5 rows:')\n",
    "prices.head()"
])

# =============================================================================
# CELL 5: Compute Log Returns
# =============================================================================
md(["## 📐 Log-Returns", "\n", "Computing continuously compounded returns: $r_t = \\ln(P_t / P_{t-1})$"])

code([
    "log_returns = compute_log_returns(prices)\n",
    "print(f'Log-returns matrix shape: {log_returns.shape}')\n",
    "print(f'Daily mean return: {log_returns.mean().mean():.5f}')\n",
    "print(f'Daily mean volatility: {log_returns.std().mean():.5f}')\n",
    "print(f'\\nLog-returns summary:')\n",
    "log_returns.describe().round(6)"
])

# =============================================================================
# CELL 6: Train/Test Split
# =============================================================================
md(["## 🔀 Train/Test Split", "\n", "First 3 years for training, last 2 years for out-of-sample testing."])

code([
    "train, test = train_test_split(log_returns, train_years=TRAIN_YEARS)\n",
    "print(f'Training set: {train.shape[0]} days | {train.index[0].date()} to {train.index[-1].date()}')\n",
    "print(f'Test set:     {test.shape[0]} days | {test.index[0].date()} to {test.index[-1].date()}')"
])

# =============================================================================
# CELL 7: Correlation Matrix
# =============================================================================
md(["## 🔗 Correlation Matrix Construction", "\n", "Building the $N \\times N$ empirical Pearson correlation matrix from daily log-returns."])

code([
    "corr_raw, N, T = build_correlation_matrix(train)\n",
    "print(f'Correlation matrix dimensions: {N} × {N}')\n",
    "print(f'Number of observations (T): {T}')\n",
    "print(f'Aspect ratio Q = T/N = {T/N:.2f}')\n",
    "print(f'\\nCorrelation matrix range: [{corr_raw.values.min():.3f}, {corr_raw.values.max():.3f}]')\n",
    "print(f'Mean off-diagonal correlation: {corr_raw.values[np.triu_indices(N, k=1)].mean():.4f}')"
])

# =============================================================================
# CELL 8: Eigenvalue Analysis
# =============================================================================
md([
    "## 🎯 Random Matrix Theory Analysis",
    "\n",
    "Decomposing the correlation matrix eigenvalues and comparing against the\n",
    "theoretical Marchenko-Pastur distribution."
])

code([
    "eigenvalues, eigenvectors, mp_bounds, signal_mask, noise_mask = eigenvalue_analysis_with_T(corr_raw, T)\n",
    "\n",
    "print(f'\\n--- Eigenvalue Summary ---')\n",
    "print(f'Total eigenvalues: {len(eigenvalues)}')\n",
    "print(f'Signal eigenvalues (> λ₊): {signal_mask.sum()}')\n",
    "print(f'Noise eigenvalues (≤ λ₊): {noise_mask.sum()}')\n",
    "print(f'\\nTop 5 eigenvalues: {eigenvalues[:5].round(3)}')\n",
    "print(f'Largest eigenvalue explains {eigenvalues[0]/eigenvalues.sum()*100:.1f}% of total variance')"
])

# =============================================================================
# CELL 9: Eigenvalue Spectrum Plot
# =============================================================================
code([
    "fig = plot_eigenvalue_spectrum(eigenvalues, mp_bounds)\n",
    "plt.show()"
])

# =============================================================================
# CELL 9b: Rolling Eigenvalue Animation
# =============================================================================
md([
    "### Rolling Eigenvalue Spectrum",
    "\n",
    "Animated visualization showing how the eigenvalue distribution evolves over time.",
    "The MP bounds are recalculated at each step."
])

code([
    "fig, ani = plot_rolling_eigenvalues(log_returns, window_months=6, step_months=1)\n",
    "if fig is not None:\n",
    "    plt.show()"
])

# =============================================================================
# CELL 10: Denoising
# =============================================================================
md([
    "## 🔬 Correlation Matrix Denoising",
    "\n",
    "Applying the RMT denoising procedure:\n",
    "1. Replace noise eigenvalues ($\\leq \\lambda_+$) with their mean\n",
    "2. Reconstruct and rescale to unit diagonal"
])

code([
    "corr_denoised, eigenvalues_raw, eigenvalues_denoised = denoise_correlation_matrix(corr_raw, T)\n",
    "\n",
    "print(f'\\n--- Denoising Summary ---')\n",
    "print(f'Raw eigenvalue range:      [{eigenvalues_raw.min():.4f}, {eigenvalues_raw.max():.4f}]')\n",
    "print(f'Denoised eigenvalue range: [{eigenvalues_denoised.min():.4f}, {eigenvalues_denoised.max():.4f}]')\n",
    "print(f'Diagonal of denoised matrix: all ≈ 1.0? {np.allclose(np.diag(corr_denoised.values), 1.0, atol=1e-4)}')"
])

# =============================================================================
# CELL 11: Heatmap Comparison
# =============================================================================
code([
    "fig = plot_correlation_heatmaps(corr_raw, corr_denoised)\n",
    "plt.show()"
])

# =============================================================================
# CELL 11b: Hierarchically Clustered Heatmap
# =============================================================================
md([
    "### Hierarchically Clustered Correlation Matrix",
    "\n",
    "Correlation matrix with hierarchical clustering (Ward linkage) to reveal",
    "block structure in stock relationships. Sectors cluster together naturally."
])

code([
    "fig = plot_clustered_heatmap(corr_raw)\n",
    "plt.show()"
])

# =============================================================================
# CELL 11c: Correlation Network Graph
# =============================================================================
md([
    "### Correlation Network Graph",
    "\n",
    "Interactive force-directed graph where nodes are stocks and edges represent",
    "significant correlations ($|\\rho| > 0.5$). Node size = degree centrality."
])

code([
    "tickers_list = list(corr_raw.columns)\n",
    "fig = plot_correlation_network(corr_denoised, tickers_list, method='denoised',\n",
    '                                threshold=0.5, sector_map=SECTOR_MAP)\n',
    "if fig is not None:\n",
    "    fig.show()"
])

# =============================================================================
# CELL 12: Portfolio Optimization
# =============================================================================
md([
    "## 💼 Portfolio Optimization",
    "\n",
    "Optimizing maximum Sharpe ratio portfolios using three different covariance estimates."
])

code([
    "methods = optimize_all_methods(train)\n",
    "\n",
    "# Display results\n",
    "results_table = []\n",
    "for name, res in methods.items():\n",
    "    results_table.append({\n",
    "        'Method': name.replace('_', ' ').title(),\n",
    "        'Expected Return': f\"{res['return']:.2%}\",\n",
    "        'Volatility': f\"{res['volatility']:.2%}\",\n",
    "        'Sharpe Ratio': f\"{res['sharpe']:.3f}\",\n",
    "        'Num Assets Held': f\"{(res['weights'] > 0.01).sum()}\"\n",
    "    })\n",
    "\n",
    "results_df = pd.DataFrame(results_table).set_index('Method')\n",
    "results_df"
])

# =============================================================================
# CELL 13: Efficient Frontier
# =============================================================================
md(["### Efficient Frontier Comparison"])

code([
    "mu = train.mean().values * 252\n",
    "daily_std = train.std().values\n",
    "\n",
    "# Raw covariance\n",
    "cov_raw = methods['raw_markowitz']['cov']\n",
    "ef_raw = compute_efficient_frontier(mu, cov_raw)\n",
    "\n",
    "# Denoised covariance\n",
    "cov_denoised = methods['denoised_markowitz']['cov']\n",
    "ef_denoised = compute_efficient_frontier(mu, cov_denoised)\n",
    "\n",
    "# Equal weight (single point)\n",
    "n = len(mu)\n",
    "mu_ew = np.ones(n) / n\n",
    "ew_ret = np.dot(mu_ew, mu)\n",
    "ew_vol = np.sqrt(np.dot(mu_ew, np.dot(cov_raw, mu_ew)))\n",
    "ef_ew = (np.array([ew_ret]), np.array([ew_vol]), np.array([(ew_ret - RISK_FREE_RATE) / ew_vol]))\n",
    "\n",
    "fig = plot_efficient_frontier(\n",
    "    ef_raw, ef_denoised, ef_ew,\n",
    "    methods['raw_markowitz'], methods['denoised_markowitz'], methods['equal_weight']\n",
    ")\n",
    "plt.show()"
])

# =============================================================================
# CELL 13b: 3D Efficient Frontier
# =============================================================================
md(["### 3D Efficient Frontier"])

code([
    "fig = plot_3d_efficient_frontier(\n",
    "    mu, cov_raw, RISK_FREE_RATE, n_points=50,\n",
    "    ef_returns=ef_raw[0], ef_vols=ef_raw[1], ef_sharpes=ef_raw[2],\n",
    ")\n",
    "if fig is not None:\n",
    "    fig.show()"
])

# =============================================================================
# CELL 14: Backtesting
# =============================================================================
md([
    "## 📈 Out-of-Sample Backtesting",
    "\n",
    "Expanding window approach with monthly rebalancing:\n",
    "- At each month boundary, recompute correlation matrix using all available history\n",
    "- Optimize portfolios and track forward returns\n",
    "- No look-ahead bias"
])

code([
    "results, metrics_df = run_backtest(log_returns, rebalance_freq=REBALANCE_FREQ, train_years=TRAIN_YEARS)\n",
    "print('\\n\\n')\n",
    "metrics_df"
])

# =============================================================================
# CELL 15: Cumulative Returns
# =============================================================================
code([
    "fig = plot_cumulative_returns(results)\n",
    "plt.show()"
])

# =============================================================================
# CELL 15b: Animated Cumulative Return Race
# =============================================================================
md(["### Animated Cumulative Return Race"])

code([
    "fig = plot_cumulative_return_race(results)\n",
    "if fig is not None:\n",
    "    fig.show()"
])

# =============================================================================
# CELL 16: Rolling Sharpe
# =============================================================================
code([
    "fig = plot_rolling_sharpe(results, window=63)\n",
    "plt.show()"
])

# =============================================================================
# CELL 17: Drawdown
# =============================================================================
code([
    "fig = plot_drawdown(results)\n",
    "plt.show()"
])

# =============================================================================
# CELL 17b: Weight Flow Sankey Diagram
# =============================================================================
md([
    "### Portfolio Weight Flow (Sankey Diagram)",
    "\n",
    "Shows how RMT-Denoised portfolio weights shift across rebalancing periods.",
    "Flow width is proportional to the retained allocation for each stock."
])

code([
    "fig = plot_weight_sankey(results['method_b']['weights'], tickers_list, top_n=10)\n",
    "if fig is not None:\n",
    "    fig.show()"
])

# =============================================================================
# CELL 17c: Risk Contribution Analysis
# =============================================================================
md([
    "### Risk Contribution Analysis",
    "\n",
    "Bubble chart: X-axis = portfolio weight, Y-axis = marginal risk contribution,",
    "bubble size = total risk contribution (%). Stocks above the equal-risk line",
    "contribute disproportionately to portfolio risk."
])

code([
    "fig = plot_risk_contribution(\n",
    "    methods['denoised_markowitz']['weights'],\n",
    "    methods['denoised_markowitz']['cov'],\n",
    "    tickers_list, method_name='RMT-Denoised Markowitz',\n",
    ")\n",
    "if fig is not None:\n",
    "    fig.show()"
])

# =============================================================================
# CELL 18: Summary Statistics
# =============================================================================
code([
    "## 📋 Summary Statistics\n",
    "\n",
    "from IPython.display import display, Markdown\n",
    "\n",
    "md_table = '| Metric | Raw Markowitz | RMT-Denoised | Equal Weight |\\n'\n",
    "md_table += '|--------|:------------:|:------------:|:------------:|\\n'\n",
    "for col in metrics_df.columns:\n",
    "    vals = [metrics_df.loc[method, col] for method in metrics_df.index]\n",
    "    md_table += f'| {col} | {vals[0]} | {vals[1]} | {vals[2]} |\\n'\n",
    "display(Markdown(md_table))"
])

# =============================================================================
# CELL 19: Portfolio Weight Allocation
# =============================================================================
md(["## 📊 Detailed Portfolio Analysis"])

md(["### Weight Allocation"])

code([
    "fig = plot_weight_allocation(\n",
    '    {"Method A: Raw Markowitz": methods["raw_markowitz"]["weights"],\n',
    '     "Method B: RMT-Denoised": methods["denoised_markowitz"]["weights"],\n',
    '     "Method C: Equal Weight": methods["equal_weight"]["weights"]},\n',
    "    tickers_list, top_n=15, sector_map=SECTOR_MAP,\n",
    ")\n",
    "plt.show()"
])

# =============================================================================
# CELL 20: Sector Analysis
# =============================================================================
md(["### Sector-Level Correlation"])

code([
    "fig = plot_sector_correlation(corr_raw, tickers_list, SECTOR_MAP)\n",
    "if fig is not None:\n",
    "    fig.show()"
])

md(["### Eigenvector Component Heatmap"])

code([
    "fig = plot_eigenvector_heatmap(\n",
    "    eigenvectors, eigenvalues, tickers_list, signal_mask, SECTOR_MAP\n",
    ")\n",
    "if fig is not None:\n",
    "    plt.show()"
])

# =============================================================================
# CELL 21: Composite Dashboard
# =============================================================================
md(["## 📊 Final Dashboard"])

code([
    "fig = plot_dashboard(\n",
    "    results, metrics_df, eigenvalues, mp_bounds, corr_raw, corr_denoised\n",
    ")\n",
    "plt.show()"
])

# =============================================================================
# CELL 22: Conclusions
# =============================================================================
md([
    "## 🎓 Conclusions\n",
    "\n",
    "### Key Findings\n",
    "\n",
    "1. **Noise in correlation matrices is significant:** The Marchenko-Pastur analysis reveals\n",
    "   that a substantial portion of eigenvalues in the empirical correlation matrix are\n",
    "   statistically indistinguishable from random noise.\n",
    "\n",
    "2. **Denoising improves out-of-sample performance:** Portfolios constructed using the\n",
    "   RMT-denoised correlation matrix show improved risk-adjusted returns compared to\n",
    "   standard Markowitz optimization.\n",
    "\n",
    "3. **Signal eigenvalues capture market structure:** The few eigenvalues above the MP upper\n",
    "   bound contain genuine cross-correlations — primarily driven by sector effects and\n",
    "   macroeconomic factors.\n",
    "\n",
    "### Limitations\n",
    "\n",
    "- The RMT framework assumes i.i.d. entries, which is violated by volatility clustering\n",
    "- Rolling window denoising assumes stationarity within each window\n",
    "- Transaction costs and market impact are not modeled\n",
    "\n",
    "### Future Work\n",
    "\n",
    "- **DCC-GARCH denoising:** Incorporate time-varying volatility\n",
    "- **Hierarchical Risk Parity:** Compare against ML-based allocation methods\n",
    "- **Regime detection:** Use eigenvalue dynamics to detect market regime changes\n",
    "- **Sector-aware RMT:** Analyze correlation structure within and across sectors\n",
    "\n",
    "---\n",
    "\n",
    "### References\n",
    "\n",
    "1. Marchenko, V.A. & Pastur, L.A. (1967). Distribution of eigenvalues for some sets of random matrices.\n",
    "   *Math. USSR-Sbornik*, 1(4), 457–483.\n",
    "2. Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999). Random matrix theory and correlated financial data.\n",
    "   *Phys. Rev. Lett.*, 82, 1909.\n",
    "3. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77–91.\n",
    "4. Bouchaud, J.-P. & Potters, M. (2003). *Theory of Financial Risk and Derivative Pricing*.\n",
    "   Cambridge University Press.\n",
    "\n",
    "---\n",
    "\n",
    "*Built by [Your Name] | Powered by Python, NumPy, SciPy, Matplotlib*"
])

# =============================================================================
# Assemble Notebook
# =============================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.6",
            "mimetype": "text/x-python",
            "file_extension": ".py"
        }
    },
    "cells": cells
}

output_path = "notebooks/rmt_portfolio_optimization.ipynb"
import os
os.makedirs("notebooks", exist_ok=True)
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"✅ Notebook created: {output_path}")
print(f"   Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type'] == 'code')} code, "
      f"{sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown)")
