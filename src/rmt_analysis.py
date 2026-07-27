"""
Random Matrix Theory Analysis
==============================
Core RMT computations: eigenvalue decomposition, Marchenko-Pastur distribution,
signal/noise separation, and correlation matrix denoising.

References
----------
- Marchenko, V.A. & Pastur, L.A. (1967). Distribution of eigenvalues for some
  sets of random matrices. Math. USSR-Sbornik, 1(4), 457–483.
- Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999). Random matrix
  theory and correlated financial data. Phys. Rev. Lett., 82, 1909.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# =============================================================================
# Correlation Matrix Construction
# =============================================================================

def build_correlation_matrix(log_returns):
    """
    Compute the empirical Pearson correlation matrix from log-returns.

    Parameters
    ----------
    log_returns : pd.DataFrame
        (T x N) daily log-returns.

    Returns
    -------
    corr_matrix : pd.DataFrame
        (N x N) correlation matrix with stock tickers as index/columns.
    N, T : int
        Dimensions (assets, observations).
    """
    corr_matrix = log_returns.corr()
    N = corr_matrix.shape[0]
    T = len(log_returns)
    return corr_matrix, N, T


# =============================================================================
# Marchenko-Pastur Distribution
# =============================================================================

def marchenko_pastur_bounds(Q, sigma2=1.0):
    """
    Compute the theoretical support bounds of the MP distribution.

    Parameters
    ----------
    Q : float
        Ratio T/N (observations / assets).
    sigma2 : float
        Variance of matrix entries (1 for correlation matrices).

    Returns
    -------
    lambda_minus, lambda_plus : float
        Lower and upper bounds of the MP eigenvalue support.
    """
    lambda_minus = sigma2 * (1 - np.sqrt(Q)) ** 2
    lambda_plus = sigma2 * (1 + np.sqrt(Q)) ** 2
    return lambda_minus, lambda_plus


def marchenko_pastur_pdf(x, Q, sigma2=1.0):
    """
    Marchenko-Pastur probability density function.

    f(λ) = (Q / 2πσ²) · sqrt((λ₊ - λ)(λ - λ₋)) / λ

    Parameters
    ----------
    x : array-like
        Eigenvalue points at which to evaluate the density.
    Q : float
        Ratio T/N.
    sigma2 : float
        Variance parameter (1 for correlation matrices).

    Returns
    -------
    density : np.ndarray
        PDF values (zero outside [λ₋, λ₊]).
    """
    x = np.asarray(x, dtype=float)
    lam_minus, lam_plus = marchenko_pastur_bounds(Q, sigma2)

    density = np.zeros_like(x)
    mask = (x >= lam_minus) & (x <= lam_plus)

    density[mask] = (
        (Q / (2 * np.pi * sigma2))
        * np.sqrt((lam_plus - x[mask]) * (x[mask] - lam_minus))
        / x[mask]
    )
    return density


# =============================================================================
# Eigenvalue Analysis
# =============================================================================

def eigenvalue_analysis(correlation_matrix):
    """
    Full eigenvalue decomposition of the correlation matrix.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame or np.ndarray
        (N x N) correlation matrix.

    Returns
    -------
    eigenvalues : np.ndarray
        Sorted eigenvalues (descending).
    eigenvectors : np.ndarray
        Corresponding eigenvectors (columns sorted by eigenvalue).
    mp_bounds : dict
        {'lambda_minus', 'lambda_plus', 'Q'}.
    signal_mask : np.ndarray[bool]
        True for eigenvalues above the MP upper bound (signal).
    noise_mask : np.ndarray[bool]
        True for eigenvalues within the MP bulk (noise).
    """
    corr = np.array(correlation_matrix, dtype=float)
    N = corr.shape[0]

    # Use eigh for symmetric matrices (faster, more stable)
    eigenvalues_raw, eigenvectors_raw = np.linalg.eigh(corr)

    # Sort descending
    idx = np.argsort(eigenvalues_raw)[::-1]
    eigenvalues = eigenvalues_raw[idx]
    eigenvectors = eigenvectors_raw[:, idx]

    # Determine Q from correlation matrix (need T)
    # For a correlation matrix of N assets, Q = T/N
    # T must be estimated or passed; here we infer from the matrix trace
    # For correlation matrices, trace = N, so Q comes from the data shape
    # We'll compute Q when we have T (pass it in or compute externally)
    # Default: use Q that makes MP fit the noise bulk
    Q = _estimate_Q(eigenvalues, N)

    lam_minus, lam_plus = marchenko_pastur_bounds(Q, sigma2=1.0)
    mp_bounds = {"lambda_minus": lam_minus, "lambda_plus": lam_plus, "Q": Q}

    signal_mask = eigenvalues > lam_plus
    noise_mask = ~signal_mask

    return eigenvalues, eigenvectors, mp_bounds, signal_mask, noise_mask


def _estimate_Q(eigenvalues, N):
    """
    Estimate Q = T/N from the eigenvalue spectrum using the MP upper bound.

    For correlation matrices, noise eigenvalues cluster around 1.
    We estimate lambda_plus from the noise bulk, then solve for Q.
    """
    sorted_eigs = np.sort(eigenvalues)
    # Bottom 70% are likely noise
    n_bottom = int(N * 0.7)
    noise_eigs = sorted_eigs[:n_bottom]
    noise_mean = noise_eigs.mean()
    noise_std = noise_eigs.std()

    # Lambda_plus ≈ mean + 2*std of the noise bulk
    lam_plus_est = noise_mean + 2 * noise_std

    # Solve: lam_plus = (1 + sqrt(Q))^2  →  Q = (sqrt(lam_plus) - 1)^2
    sqrt_q = np.sqrt(max(lam_plus_est, 1.01)) - 1
    Q = max(sqrt_q ** 2, 1.0)

    return Q


def eigenvalue_analysis_with_T(correlation_matrix, T):
    """
    Eigenvalue analysis with known T (number of observations).

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
    T : int
        Number of trading days (observations).

    Returns
    -------
    Same as eigenvalue_analysis but with correct Q = T/N.
    """
    corr = np.array(correlation_matrix, dtype=float)
    N = corr.shape[0]
    Q = T / N

    eigenvalues_raw, eigenvectors_raw = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues_raw)[::-1]
    eigenvalues = eigenvalues_raw[idx]
    eigenvectors = eigenvectors_raw[:, idx]

    lam_minus, lam_plus = marchenko_pastur_bounds(Q, sigma2=1.0)
    mp_bounds = {"lambda_minus": lam_minus, "lambda_plus": lam_plus, "Q": Q}

    signal_mask = eigenvalues > lam_plus
    noise_mask = ~signal_mask

    n_signal = signal_mask.sum()
    total_var = eigenvalues.sum()
    signal_var = eigenvalues[signal_mask].sum()
    print(f"[RMT] Q = T/N = {T}/{N} = {Q:.2f}")
    print(f"[RMT] MP bounds: λ₋ = {lam_minus:.4f}, λ₊ = {lam_plus:.4f}")
    print(f"[RMT] Signal eigenvalues: {n_signal} / {N}")
    print(f"[RMT] Signal explains {signal_var/total_var*100:.1f}% of total variance")

    return eigenvalues, eigenvectors, mp_bounds, signal_mask, noise_mask


# =============================================================================
# Correlation Matrix Denoising
# =============================================================================

def denoise_correlation_matrix(correlation_matrix, T, method="constant_rescale"):
    """
    Denoise a correlation matrix by replacing noise eigenvalues with their mean.

    Procedure (Laloux et al. 1999):
    1. Eigendecompose: C = V Λ Vᵀ
    2. Replace noise eigenvalues (≤ λ₊) with their average
    3. Reconstruct: C_denoised = V Λ_denoised Vᵀ
    4. Rescale to unit diagonal: C̃ = D^{-½} C_denoised D^{-½}

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
    T : int
        Number of observations.
    method : str
        'constant_rescale' — standard RMT denoising.

    Returns
    -------
    denoised : pd.DataFrame
        Denoised correlation matrix (same shape/index as input).
    eigenvalues_raw, eigenvalues_denoised : np.ndarray
        For comparison.
    """
    corr = np.array(correlation_matrix, dtype=float)
    N = corr.shape[0]
    Q = T / N

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # MP bounds
    lam_minus, lam_plus = marchenko_pastur_bounds(Q, sigma2=1.0)

    # Identify noise eigenvalues
    noise_mask = eigenvalues <= lam_plus
    n_noise = noise_mask.sum()
    n_signal = N - n_noise

    # Replace noise eigenvalues with their mean
    eigenvalues_denoised = eigenvalues.copy()
    if n_noise > 0:
        mean_noise = eigenvalues[noise_mask].mean()
        eigenvalues_denoised[noise_mask] = mean_noise

    # Reconstruct correlation matrix
    C_denoised = eigenvectors @ np.diag(eigenvalues_denoised) @ eigenvectors.T

    # Rescale to unit diagonal
    d = np.sqrt(np.diag(C_denoised))
    d = np.where(d > 1e-10, d, 1.0)  # Avoid division by zero
    D_inv = np.diag(1.0 / d)
    C_final = D_inv @ C_denoised @ D_inv

    # Small regularization for numerical stability
    C_final += 1e-6 * np.eye(N)

    # Convert to DataFrame
    tickers = correlation_matrix.index if hasattr(correlation_matrix, 'index') else None
    denoised = pd.DataFrame(C_final, index=tickers, columns=tickers)

    print(f"[RMT] Denoised: replaced {n_noise} noise eigenvalues (mean={mean_noise:.4f})")
    print(f"[RMT] Kept {n_signal} signal eigenvalues")

    return denoised, eigenvalues, eigenvalues_denoised


# =============================================================================
# Participation Ratio
# =============================================================================

def participation_ratio(eigenvectors):
    """
    Compute the participation ratio for each eigenvector.
    PR ≈ 1 / Σ(v_i⁴). Measures localization:
      - PR ≈ 1  → eigenvector concentrated on one stock (idiosyncratic)
      - PR ≈ N  → eigenvector delocalized (market-wide factor)

    Parameters
    ----------
    eigenvectors : np.ndarray
        (N x K) matrix of eigenvectors (columns).

    Returns
    -------
    pr : np.ndarray
        Participation ratio for each eigenvector.
    """
    pr = 1.0 / np.sum(eigenvectors ** 4, axis=0)
    return pr


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    from data_fetcher import fetch_stock_data, compute_log_returns

    prices = fetch_stock_data()
    log_ret = compute_log_returns(prices)
    corr, N, T = build_correlation_matrix(log_ret)

    eigenvalues, eigvecs, mp_bounds, sig_mask, noise_mask = eigenvalue_analysis_with_T(corr, T)
    denoised, _, _ = denoise_correlation_matrix(corr, T)

    print(f"\nTop 5 eigenvalues: {eigenvalues[:5].round(3)}")
    print(f"MP upper bound: {mp_bounds['lambda_plus']:.3f}")
    print(f"Signal eigenvalues: {sig_mask.sum()}")
