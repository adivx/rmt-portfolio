"""
Configuration for RMT Portfolio Optimization Project
=====================================================
Centralized constants for data, colors, and analysis parameters.
"""

# =============================================================================
# NIFTY 50 Stock Tickers (NSE, .NS suffix for yfinance)
# =============================================================================
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS",
    "TATAMOTORS.NS", "HCLTECH.NS", "ONGC.NS", "NTPC.NS", "TATASTEEL.NS",
    "POWERGRID.NS", "M&M.NS", "BAJAJFINSV.NS", "TECHM.NS", "INDUSINDBK.NS",
    "HINDALCO.NS", "GRASIM.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS",
    "EICHERMOT.NS", "APOLLOHOSP.NS", "COALINDIA.NS", "BRITANNIA.NS",
    "TATACONSUM.NS", "SBILIFE.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "LTIM.NS", "HDFCLIFE.NS", "BEL.NS",
    "TRENT.NS", "IRCTC.NS",
]

# =============================================================================
# Dark Mode Color Palette (Neon on Dark)
# =============================================================================
COLORS = {
    # Background layers
    "bg_dark":       "#0d0d0d",
    "surface":       "#1a1a19",
    "surface_light": "#252523",

    # Text hierarchy
    "text_primary":   "#ffffff",
    "text_secondary": "#c3c2b7",
    "text_muted":     "#898781",

    # Grid / Baselines
    "grid":     "#2c2c2a",
    "baseline": "#383835",

    # Primary series (3 portfolio methods)
    "method_a": "#ff006e",  # Magenta — Raw Markowitz
    "method_b": "#00d4ff",  # Cyan    — RMT-Denoised
    "method_c": "#39ff14",  # Green   — Equal Weight

    # Supporting accents
    "gold":   "#ffd700",  # MP distribution curve
    "purple": "#b388ff",
    "orange": "#ff6b35",

    # Eigenvalue highlights
    "signal": "#00d4ff",  # Cyan
    "noise":  "#ff006e",  # Magenta (low alpha in plots)

    # Heatmap diverging: dark_blue -> surface -> cyan
    "hm_neg": "#0d366b",
    "hm_zero": "#1a1a19",
    "hm_pos": "#00d4ff",
}

# =============================================================================
# Data Parameters
# =============================================================================
START_DATE      = "2019-01-01"
END_DATE        = "2024-12-31"
TRAIN_YEARS     = 3        # First 3 years for training
TEST_YEARS      = 2        # Last 2 years for testing
RISK_FREE_RATE  = 0.07     # India 10Y Government Bond yield (annualized)
REBALANCE_FREQ  = "M"      # Monthly rebalancing
MIN_STOCKS      = 35       # Minimum stocks needed for statistical validity
DATA_CACHE_DIR  = "data/"
OUTPUT_DIR      = "output/"
FIGURES_DIR     = "figures/"

# =============================================================================
# Sector Mapping for NIFTY 50 Stocks
# =============================================================================
SECTOR_MAP = {
    "RELIANCE.NS": "Oil & Gas",
    "TCS.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "INFY.NS": "IT",
    "ICICIBANK.NS": "Banking",
    "HINDUNILVR.NS": "FMCG",
    "SBIN.NS": "Banking",
    "BHARTIARTL.NS": "Telecom",
    "ITC.NS": "FMCG",
    "KOTAKBANK.NS": "Banking",
    "LT.NS": "Infrastructure",
    "AXISBANK.NS": "Banking",
    "BAJFINANCE.NS": "Financial Services",
    "ASIANPAINT.NS": "Consumer Durables",
    "MARUTI.NS": "Automobile",
    "TITAN.NS": "Consumer Durables",
    "SUNPHARMA.NS": "Pharma",
    "ULTRACEMCO.NS": "Cement",
    "NESTLEIND.NS": "FMCG",
    "WIPRO.NS": "IT",
    "TATAMOTORS.NS": "Automobile",
    "HCLTECH.NS": "IT",
    "ONGC.NS": "Oil & Gas",
    "NTPC.NS": "Power",
    "TATASTEEL.NS": "Metals",
    "POWERGRID.NS": "Power",
    "M&M.NS": "Automobile",
    "BAJAJFINSV.NS": "Financial Services",
    "TECHM.NS": "IT",
    "INDUSINDBK.NS": "Banking",
    "HINDALCO.NS": "Metals",
    "GRASIM.NS": "Cement",
    "DIVISLAB.NS": "Pharma",
    "DRREDDY.NS": "Pharma",
    "CIPLA.NS": "Pharma",
    "EICHERMOT.NS": "Automobile",
    "APOLLOHOSP.NS": "Healthcare",
    "COALINDIA.NS": "Mining",
    "BRITANNIA.NS": "FMCG",
    "TATACONSUM.NS": "FMCG",
    "SBILIFE.NS": "Insurance",
    "HEROMOTOCO.NS": "Automobile",
    "BAJAJ-AUTO.NS": "Automobile",
    "ADANIENT.NS": "Conglomerate",
    "ADANIPORTS.NS": "Infrastructure",
    "LTIM.NS": "IT",
    "HDFCLIFE.NS": "Insurance",
    "BEL.NS": "Defence",
    "TRENT.NS": "Retail",
    "IRCTC.NS": "Services",
}

# =============================================================================
# Sector Colors (consistent across all visualizations)
# =============================================================================
SECTOR_COLORS = {
    "Banking":            "#00d4ff",
    "IT":                 "#39ff14",
    "FMCG":               "#ff006e",
    "Oil & Gas":          "#ffd700",
    "Automobile":         "#ff6b35",
    "Pharma":             "#b388ff",
    "Financial Services": "#00ff88",
    "Infrastructure":     "#ff4444",
    "Metals":             "#888888",
    "Power":              "#ffaa00",
    "Cement":             "#8B4513",
    "Telecom":            "#4488ff",
    "Consumer Durables":  "#ff69b4",
    "Insurance":          "#00CED1",
    "Healthcare":         "#98FB98",
    "Mining":             "#D2691E",
    "Conglomerate":       "#DDA0DD",
    "Defence":            "#228B22",
    "Retail":             "#FF6347",
    "Services":           "#4682B4",
}
