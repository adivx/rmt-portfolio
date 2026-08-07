# 📊 Random Matrix Theory for Indian Stock Market Portfolio Optimization

> Applying Marchenko-Pastur theory to denoise correlation matrices for improved portfolio construction on NIFTY 50 stocks.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]()
[![NSE Data](https://img.shields.io/badge/Data-NSE%20NIFTY%2050-blue?style=for-the-badge)]()

---

## 🎯 Overview

**Problem:** Standard Markowitz mean-variance optimization is notoriously sensitive to estimation errors in the covariance matrix. When the number of assets ($N$) is comparable to the number of observations ($T$), the empirical correlation matrix contains significant noise that degrades portfolio performance.

**Solution:** Random Matrix Theory (RMT) provides a rigorous mathematical framework to separate signal from noise in correlation matrices. By filtering out eigenvalues that are statistically indistinguishable from random noise (using the Marchenko-Pastur distribution as null hypothesis), we construct more robust portfolios.

**Key Result:** RMT-denoised portfolios demonstrate superior out-of-sample risk-adjusted returns compared to traditional Markowitz optimization on NIFTY 50 stocks (2019–2024).

---

## 📐 Mathematical Framework

### Marchenko-Pastur Distribution

For an $N \times T$ random matrix with i.i.d. entries of variance $\sigma^2/T$, the eigenvalue density converges to:

$$f_{MP}(\lambda) = \frac{Q}{2\pi\sigma^2} \cdot \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{\lambda}$$

where $Q = T/N$ and the support bounds are:

$$\lambda_{\pm} = \sigma^2\left(1 \pm \sqrt{Q}\right)^2$$

### Denoising Procedure

1. Eigendecompose: $C = V \Lambda V^\top$
2. Replace noise eigenvalues ($\leq \lambda_+$) with their mean
3. Reconstruct: $\tilde{C} = V \tilde{\Lambda} V^\top$
4. Rescale to unit diagonal: $\hat{C} = D^{-1/2} \tilde{C} D^{-1/2}$

---

## 📊 Results

| Method | Sharpe Ratio | Max Drawdown | Annual Return |
|--------|:-----------:|:------------:|:-------------:|
| **Raw Markowitz** | Baseline | Baseline | Baseline |
| **RMT-Denoised** | ✅ Improved | ✅ Reduced | ✅ Competitive |
| **Equal Weight (1/N)** | Reference | Reference | Reference |

### Visualizations

| Eigenvalue Spectrum | Correlation Heatmaps | Efficient Frontier |
|:-------------------:|:-------------------:|:------------------:|
| *MP distribution overlay* | *Raw vs Denoised* | *3-method comparison* |

| Cumulative Returns | Rolling Sharpe | Drawdown |
|:------------------:|:--------------:|:--------:|
| *Out-of-sample performance* | *Time-varying risk-adjusted returns* | *Risk analysis* |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/adivx/rmt-portfolio.git
cd rmt-portfolio
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook notebooks/rmt_portfolio_optimization.ipynb
```

### 4. Or Run as Python Script

```bash
python src/backtester.py
```

---

## 📁 Project Structure

```
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── config.py                              # Global configuration
├── create_notebook.py                     # Notebook generator script
│
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py                    # NSE data download & preprocessing
│   ├── rmt_analysis.py                    # Core RMT: MP distribution, denoising
│   ├── portfolio_optimizer.py             # Markowitz optimization (3 methods)
│   ├── backtester.py                      # Out-of-sample backtesting engine
│   └── visualization.py                   # Dark mode visualization suite
│
├── notebooks/
│   └── rmt_portfolio_optimization.ipynb   # Main presentation notebook
│
└── data/                                  # Cached stock data (auto-generated)
```

---

## 🔧 Configuration

All parameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NIFTY50_TICKERS` | 50 stocks | NSE stock universe |
| `START_DATE` | 2019-01-01 | Data start date |
| `END_DATE` | 2024-12-31 | Data end date |
| `TRAIN_YEARS` | 3 | Training window |
| `RISK_FREE_RATE` | 7% | India 10Y bond yield |
| `REBALANCE_FREQ` | Monthly | Portfolio rebalancing frequency |

---

## 📦 Dependencies

- `yfinance` — Yahoo Finance data API
- `numpy` — Numerical computing
- `scipy` — Scientific computing & optimization
- `pandas` — Data manipulation
- `matplotlib` — Visualization
- `seaborn` — Statistical visualization

---

## 📚 References

1. **Marchenko, V.A. & Pastur, L.A.** (1967). Distribution of eigenvalues for some sets of random matrices. *Math. USSR-Sbornik*, 1(4), 457–483.
2. **Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M.** (1999). Random matrix theory and correlated financial data. *Phys. Rev. Lett.*, 82, 1909.
3. **Markowitz, H.** (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77–91.
4. **Bouchaud, J.-P. & Potters, M.** (2003). *Theory of Financial Risk and Derivative Pricing*. Cambridge University Press.

---

## 👤 Author

**Aditya Kumar**
- GitHub: [@adivx](https://github.com/adivx)
- LinkedIn: [Aditya Kumar](https://www.linkedin.com/in/aditya-kumar-49b976383/)
- Email: aditya.kumar.x9182@gmail.com

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using Python, NumPy, SciPy, and Matplotlib*
