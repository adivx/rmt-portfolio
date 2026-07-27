"""
Visualization Suite — Neon-on-Dark Charts for RMT Portfolio Analysis
====================================================================
16 publication-quality visualizations + 3 glow/gradient helper functions.
Matplotlib for static charts, Plotly for interactive ones.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from matplotlib import patheffects
import seaborn as sns

from config import COLORS, RISK_FREE_RATE

# Plotly imports (lazy — only needed for interactive charts)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# NetworkX imports (lazy)
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# scipy for clustering
try:
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import squareform
    HAS_SCIPY_CLUSTER = True
except ImportError:
    HAS_SCIPY_CLUSTER = False


# =============================================================================
# Global Dark Mode Setup
# =============================================================================

def setup_dark_style():
    """Configure matplotlib for dark mode with neon accents and glow effects."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor":    COLORS["bg_dark"],
        "axes.facecolor":      COLORS["surface"],
        "axes.edgecolor":      COLORS["baseline"],
        "axes.labelcolor":     COLORS["text_primary"],
        "text.color":          COLORS["text_primary"],
        "xtick.color":         COLORS["text_secondary"],
        "ytick.color":         COLORS["text_secondary"],
        "grid.color":          COLORS["grid"],
        "grid.linewidth":      0.5,
        "font.family":         "sans-serif",
        "font.size":           11,
        "axes.titlesize":      14,
        "axes.titleweight":    "bold",
        "figure.titlesize":    18,
        "legend.fontsize":     10,
        "savefig.dpi":         200,
        "savefig.facecolor":   COLORS["bg_dark"],
        "savefig.bbox":        "tight",
        "axes.prop_cycle":     plt.cycler(color=[
            COLORS["method_b"], COLORS["method_a"], COLORS["method_c"],
            COLORS["gold"], COLORS["purple"], COLORS["orange"],
        ]),
    })


# =============================================================================
# Visual Effect Helpers
# =============================================================================

def add_glow(ax, color, alpha=0.12, linewidth=3, n_layers=6):
    """Add a neon glow effect to the last plotted line on an axes."""
    lines = ax.get_lines()
    if not lines:
        return
    line = lines[-1]
    x, y = line.get_data()
    for i in range(n_layers, 0, -1):
        ax.plot(x, y, color=color, linewidth=linewidth + i * 2.5,
                alpha=alpha / n_layers, solid_capstyle="round",
                zorder=line.get_zorder() - 1)


def gradient_fill(ax, x, y, y_baseline, color, alpha_max=0.3, n_gradient=30):
    """Fill between with a smooth vertical gradient effect."""
    for i in range(n_gradient):
        a = alpha_max * (1 - i / n_gradient) ** 2
        frac_i = i / n_gradient
        frac_j = (i + 1) / n_gradient
        y_low = y_baseline + (y - y_baseline) * frac_i
        y_high = y_baseline + (y - y_baseline) * frac_j
        ax.fill_between(x, y_low, y_high, color=color, alpha=a, linewidth=0)


def glow_scatter(ax, x, y, color, size=200, label=None, zorder=5):
    """Scatter point with outer glow halo effect."""
    ax.scatter(x, y, s=size * 4, color=color, alpha=0.1, zorder=zorder - 1)
    ax.scatter(x, y, s=size * 2, color=color, alpha=0.2, zorder=zorder - 1)
    ax.scatter(x, y, s=size, color=color, edgecolors="white",
               linewidths=1.5, zorder=zorder, label=label)


def _frosted_box(ax, text, x, y, fontsize=10, color=COLORS["text_primary"],
                 bg_color=COLORS["surface"], alpha=0.85):
    """Draw a frosted-glass text annotation box."""
    txt = ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
                  color=color, fontfamily="monospace", va="top", ha="left",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color,
                            edgecolor=COLORS["baseline"], alpha=alpha))
    txt.set_path_effects([patheffects.withSimplePatchShadow(offset=(1, -1),
                          shadow_rgbFace=COLORS["bg_dark"], alpha=0.4)])
    return txt


# =============================================================================
# Chart 1: Eigenvalue Distribution with MP Overlay (Enhanced)
# =============================================================================

def plot_eigenvalue_spectrum(eigenvalues, mp_bounds, T=None, save_path=None):
    """
    Histogram of empirical eigenvalues overlaid with the MP theoretical curve.
    Enhanced with gradient fill, glow effects, signal variance inset, and
    particle dots at signal eigenvalue positions.
    """
    Q = mp_bounds["Q"]
    lam_minus = mp_bounds["lambda_minus"]
    lam_plus = mp_bounds["lambda_plus"]

    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 1, 1.2], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_pie = fig.add_subplot(gs[1])
    ax_stats = fig.add_subplot(gs[2])

    # --- Main histogram with gradient fill ---
    n_bins = min(50, len(eigenvalues))
    counts, bins, patches = ax.hist(
        eigenvalues, bins=n_bins, density=True, alpha=0.35,
        color=COLORS["method_b"], edgecolor="none",
    )
    bin_centers = 0.5 * (bins[:-1] + bins[1])
    gradient_fill(ax, bin_centers, counts, 0, COLORS["method_b"], alpha_max=0.35)

    # --- MP theoretical curve with glow ---
    x_mp = np.linspace(max(lam_minus - 0.1, 0), lam_plus + 0.8, 500)
    from src.rmt_analysis import marchenko_pastur_pdf
    y_mp = marchenko_pastur_pdf(x_mp, Q, sigma2=1.0)
    ax.plot(x_mp, y_mp, color=COLORS["gold"], linewidth=2.5, zorder=5,
            label=f"MP (Q={Q:.1f})")
    add_glow(ax, COLORS["gold"], linewidth=2.5, n_layers=5)

    # --- MP bounds ---
    ax.axvline(lam_plus, color=COLORS["method_a"], linestyle="--", linewidth=1.5,
               label=f"$\\lambda_+$ = {lam_plus:.3f}", alpha=0.9)
    ax.axvline(lam_minus, color=COLORS["purple"], linestyle="--", linewidth=1.5,
               label=f"$\\lambda_-$ = {lam_minus:.3f}", alpha=0.7)

    # --- Shade noise region ---
    ax.axvspan(lam_minus, lam_plus, alpha=0.06, color=COLORS["method_a"],
               label="Noise bulk")

    # --- Signal eigenvalue particle dots ---
    signal_mask = eigenvalues > lam_plus
    signal_eigs = eigenvalues[signal_mask]
    if len(signal_eigs) > 0:
        for i, eig in enumerate(signal_eigs[:8]):
            glow_scatter(ax, eig, 0, COLORS["method_b"], size=80,
                         label="Signal eigenvalues" if i == 0 else None)

    ax.set_xlabel("Eigenvalue ($\\lambda$)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Eigenvalue Spectrum vs Marchenko-Pastur Distribution",
                 fontsize=15, pad=12)
    ax.legend(loc="upper right", framealpha=0.3, edgecolor=COLORS["baseline"],
              fontsize=9)
    ax.grid(True, alpha=0.2)

    # --- Inset: Signal vs Noise variance pie ---
    total_var = eigenvalues.sum()
    noise_var = eigenvalues[~signal_mask].sum() if not signal_mask.all() else 0
    signal_var = eigenvalues[signal_mask].sum() if signal_mask.any() else 0
    n_signal = signal_mask.sum()
    n_noise = (~signal_mask).sum()

    sizes = [signal_var, noise_var]
    labels_pie = [f"Signal\n({n_signal} eig)", f"Noise\n({n_noise} eig)"]
    colors_pie = [COLORS["method_b"], COLORS["method_a"]]
    explode = (0.05, 0)
    ax_pie.pie(sizes, labels=labels_pie, colors=colors_pie, explode=explode,
               autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8, "color": "white"})
    ax_pie.set_title("Variance", fontsize=10, pad=5)

    # --- Stats panel ---
    ax_stats.axis("off")
    stats_lines = [
        f"Total eigenvalues: {len(eigenvalues)}",
        f"Signal: {n_signal} ({n_signal/len(eigenvalues)*100:.0f}%)",
        f"Noise:  {n_noise} ({n_noise/len(eigenvalues)*100:.0f}%)",
        f"",
        f"$\\lambda_{{max}}$ = {eigenvalues[0]:.3f}",
        f"Explains {eigenvalues[0]/total_var*100:.1f}% variance",
        f"",
        f"Q = T/N = {Q:.2f}",
        f"$\\lambda_+$ = {lam_plus:.3f}",
        f"$\\lambda_-$ = {lam_minus:.3f}",
    ]
    if T is not None:
        stats_lines.insert(3, f"T = {T} days")
    _frosted_box(ax_stats, "\n".join(stats_lines), 0.05, 0.95, fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# Chart 2: Raw vs Denoised Correlation Heatmaps (Enhanced)
# =============================================================================

def plot_correlation_heatmaps(corr_raw, corr_denoised, save_path=None):
    """Side-by-side heatmaps of raw and denoised correlation matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    cmap = LinearSegmentedColormap.from_list(
        "rmt_cmap", [COLORS["hm_neg"], COLORS["hm_zero"], COLORS["hm_pos"]]
    )

    n_show = min(20, corr_raw.shape[0])
    tickers = corr_raw.columns[:n_show]

    for ax, data, title in zip(
        axes,
        [corr_raw.iloc[:n_show, :n_show], corr_denoised.iloc[:n_show, :n_show]],
        ["Raw Correlation Matrix", "RMT-Denoised Correlation Matrix"],
    ):
        im = ax.imshow(data.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_show))
        ax.set_yticks(range(n_show))
        ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(tickers, fontsize=7)
        ax.set_title(title, fontsize=13, pad=10)

    # Colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Correlation", color=COLORS["text_primary"])
    cbar.ax.yaxis.set_tick_params(color=COLORS["text_primary"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=COLORS["text_primary"])

    # Info annotation
    mean_raw = corr_raw.values[np.triu_indices(n_show, k=1)].mean()
    mean_denoised = corr_denoised.values[np.triu_indices(n_show, k=1)].mean()
    _frosted_box(axes[0], f"Mean off-diag: {mean_raw:.3f}", 0.02, 0.02, fontsize=9)
    _frosted_box(axes[1], f"Mean off-diag: {mean_denoised:.3f}", 0.02, 0.02, fontsize=9)

    plt.suptitle("Raw vs RMT-Denoised Correlation Matrices", fontsize=16, y=1.02,
                 color=COLORS["text_primary"])
    plt.tight_layout(rect=[0, 0, 0.92, 1])

    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# Chart 3: Efficient Frontier Comparison (Enhanced)
# =============================================================================

def plot_efficient_frontier(
    ef_raw, ef_denoised, ef_ew,
    opt_raw, opt_denoised, opt_ew,
    rf=RISK_FREE_RATE,
    save_path=None,
):
    """Three efficient frontier curves with optimal Sharpe points and glow effects."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Frontier curves with gradient fills
    for (r, v, s), color, label in [
        (ef_raw, COLORS["method_a"], "Raw Markowitz"),
        (ef_denoised, COLORS["method_b"], "RMT-Denoised Markowitz"),
        (ef_ew, COLORS["method_c"], "Equal Weight"),
    ]:
        if len(r) > 0:
            # Sort by volatility for clean curve
            idx = np.argsort(v)
            ax.plot(v[idx], r[idx], color=color, linewidth=2.2, label=label, alpha=0.9)
            add_glow(ax, color, linewidth=2.2, n_layers=4)
            # Gradient fill under curve
            gradient_fill(ax, v[idx], r[idx], rf, color, alpha_max=0.08, n_gradient=15)

    # Optimal Sharpe points with glow
    for opt, color, label in [
        (opt_raw, COLORS["method_a"], "A: Raw"),
        (opt_denoised, COLORS["method_b"], "B: RMT-Denoised"),
        (opt_ew, COLORS["method_c"], "C: Equal Weight"),
    ]:
        glow_scatter(ax, opt["volatility"], opt["return"], color, size=180,
                     label=f"Optimal {label}")

    # Risk-free rate line
    ax.axhline(rf, color=COLORS["gold"], linestyle=":", linewidth=1.5,
               label=f"Risk-Free Rate ({rf:.1%})", alpha=0.7)

    # Sharpe annotation lines from optimal points to axes
    for opt, color in [(opt_denoised, COLORS["method_b"])]:
        sharpe_line_x = np.linspace(0, opt["volatility"], 50)
        sharpe_line_y = rf + sharpe_line_x * opt["sharpe"]
        ax.plot(sharpe_line_x, sharpe_line_y, color=color, linewidth=0.8,
                linestyle="--", alpha=0.4)

    ax.set_xlabel("Annualized Volatility", fontsize=12)
    ax.set_ylabel("Annualized Return", fontsize=12)
    ax.set_title("Efficient Frontier: Raw vs RMT-Denoised vs Equal Weight",
                 fontsize=15, pad=12)
    ax.legend(loc="upper left", framealpha=0.3, edgecolor=COLORS["baseline"], fontsize=9)
    ax.grid(True, alpha=0.2)

    # Info box
    info = (f"Best Sharpe: RMT-Denoised = {opt_denoised['sharpe']:.3f}\n"
            f"Raw Markowitz = {opt_raw['sharpe']:.3f}\n"
            f"Improvement: {(opt_denoised['sharpe'] - opt_raw['sharpe'])/abs(opt_raw['sharpe'])*100:+.1f}%")
    _frosted_box(ax, info, 0.55, 0.95, fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# Chart 4: Cumulative Return Curves (Enhanced)
# =============================================================================

def plot_cumulative_returns(results, save_path=None):
    """Plot cumulative returns for all three methods with gradient fills."""
    fig, ax = plt.subplots(figsize=(14, 6))

    methods = [
        ("method_a", COLORS["method_a"], "Method A: Raw Markowitz"),
        ("method_b", COLORS["method_b"], "Method B: RMT-Denoised"),
        ("method_c", COLORS["method_c"], "Method C: Equal Weight (1/N)"),
    ]

    max_val = 1.0
    for key, color, label in methods:
        cum = results[key]["cumulative"]
        if len(cum) > 0:
            cum_norm = cum / cum.iloc[0]
            max_val = max(max_val, cum_norm.max())
            ax.plot(cum_norm.index, cum_norm.values, color=color, linewidth=2,
                    label=label, alpha=0.9)
            add_glow(ax, color, linewidth=2, n_layers=3)
            gradient_fill(ax, cum_norm.index, cum_norm.values, 1.0, color,
                          alpha_max=0.06, n_gradient=15)

    # Reference line at 1.0
    ax.axhline(1.0, color=COLORS["text_muted"], linestyle=":", linewidth=1, alpha=0.5)

    # Highlight best performer
    best_key = max(methods, key=lambda m: (
        results[m[0]]["cumulative"].iloc[-1] / results[m[0]]["cumulative"].iloc[0]
        if len(results[m[0]]["cumulative"]) > 0 else 0
    ))
    best_cum = results[best_key[0]]["cumulative"]
    if len(best_cum) > 0:
        best_norm = best_cum / best_cum.iloc[0]
        glow_scatter(ax, best_norm.index[-1], best_norm.values[-1], best_key[1],
                     size=150, label=f"Best: {best_key[2].split(': ')[1]}")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value (Normalized to 1.0)", fontsize=12)
    ax.set_title("Out-of-Sample Cumulative Returns", fontsize=15, pad=12)
    ax.legend(loc="upper left", framealpha=0.3, edgecolor=COLORS["baseline"])
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# Chart 5: Rolling Sharpe Ratio (Enhanced)
# =============================================================================

def plot_rolling_sharpe(results, window=63, rf=RISK_FREE_RATE, save_path=None):
    """Plot rolling Sharpe ratio with gradient fills and volatility-of-volatility band."""
    fig, ax = plt.subplots(figsize=(14, 5))

    methods = [
        ("method_a", COLORS["method_a"], "Raw Markowitz"),
        ("method_b", COLORS["method_b"], "RMT-Denoised"),
        ("method_c", COLORS["method_c"], "Equal Weight"),
    ]

    for key, color, label in methods:
        daily = results[key]["daily"]
        if len(daily) > window:
            rolling_ret = daily.rolling(window).mean() * 252
            rolling_vol = daily.rolling(window).std() * np.sqrt(252)
            rolling_sharpe = (rolling_ret - rf) / rolling_vol

            rs_clean = rolling_sharpe.dropna()
            if len(rs_clean) > 0:
                ax.plot(rs_clean.index, rs_clean.values, color=color, linewidth=1.5,
                        label=label, alpha=0.85)
                add_glow(ax, color, linewidth=1.5, n_layers=2)

                # Vol-of-vol band (±1 std of rolling Sharpe)
                rolling_std = rs_clean.rolling(window).std()
                upper = rs_clean + rolling_std
                lower = rs_clean - rolling_std
                ax.fill_between(rs_clean.index, lower.values, upper.values,
                                color=color, alpha=0.06)

    ax.axhline(0, color=COLORS["text_muted"], linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_title(f"Rolling Sharpe Ratio ({window}-day window)", fontsize=15, pad=12)
    ax.legend(loc="upper left", framealpha=0.3, edgecolor=COLORS["baseline"])
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# Chart 6: Drawdown Comparison (Enhanced — Underwater Style)
# =============================================================================

def plot_drawdown(results, save_path=None):
    """Plot drawdown comparison with gradient depth fills (underwater chart)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    methods = [
        ("method_a", COLORS["method_a"], "Raw Markowitz"),
        ("method_b", COLORS["method_b"], "RMT-Denoised"),
        ("method_c", COLORS["method_c"], "Equal Weight"),
    ]

    for key, color, label in methods:
        dd = results[key]["drawdown"]
        if len(dd) > 0:
            dd_pct = dd.values * 100
            # Gradient depth fill
            gradient_fill(ax, dd.index, dd_pct, 0, color, alpha_max=0.15, n_gradient=20)
            ax.plot(dd.index, dd_pct, color=color, linewidth=1.5, label=label, alpha=0.9)

            # Annotate worst drawdown with glow
            worst_idx = dd.idxmin()
            worst_val = dd.min() * 100
            glow_scatter(ax, worst_idx, worst_val, color, size=120)
            ax.annotate(
                f"{worst_val:.1f}%",
                xy=(worst_idx, worst_val),
                xytext=(0, -18), textcoords="offset points",
                fontsize=9, fontweight="bold", color=color,
                ha="center",
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.5),
            )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_title("Portfolio Drawdown Analysis (Underwater Chart)", fontsize=15, pad=12)
    ax.legend(loc="lower left", framealpha=0.3, edgecolor=COLORS["baseline"])
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# Chart 7: Weight Allocation — Lollipop Style (Enhanced)
# =============================================================================

def plot_weight_allocation(weights_dict, tickers, top_n=15, sector_map=None,
                           save_path=None):
    """Horizontal lollipop chart of portfolio weights with sector color coding."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    methods = [
        ("Method A: Raw Markowitz", COLORS["method_a"]),
        ("Method B: RMT-Denoised", COLORS["method_b"]),
        ("Method C: Equal Weight", COLORS["method_c"]),
    ]

    for ax, (name, default_color) in zip(axes, methods):
        w = weights_dict[name]
        idx = np.argsort(w)[::-1][:top_n]
        top_tickers = [tickers[i] for i in idx]
        top_weights = w[idx] * 100

        # Color by sector if available (resolve sector name → color)
        if sector_map:
            from config import SECTOR_COLORS
            bar_colors = [SECTOR_COLORS.get(sector_map.get(t, ""), default_color)
                          for t in top_tickers]
        else:
            bar_colors = default_color

        y_pos = range(len(top_tickers))
        # Stem lines
        ax.hlines(y_pos, 0, top_weights, color=bar_colors, linewidth=1.5, alpha=0.7)
        # Lollipop heads
        ax.scatter(top_weights, y_pos, color=bar_colors, s=60, zorder=5,
                   edgecolors="white", linewidths=0.5)
        # Weight labels
        for i, (w_val, y) in enumerate(zip(top_weights, y_pos)):
            if w_val > 0.5:
                ax.text(w_val + 0.3, y, f"{w_val:.1f}%", va="center",
                        fontsize=7, color=COLORS["text_secondary"])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_tickers, fontsize=8)
        ax.set_xlabel("Weight (%)", fontsize=10)
        ax.set_title(name, fontsize=12, color=default_color, pad=10)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.2)
        ax.set_xlim(0, max(top_weights) * 1.15 if len(top_weights) > 0 else 10)

    plt.suptitle("Portfolio Weight Allocation (Top 15 Stocks)", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# Chart 8: Correlation Network (Plotly Interactive)
# =============================================================================

def plot_correlation_network(corr_matrix, tickers, method="denoised",
                             threshold=0.5, sector_map=None, save_path=None):
    """
    Interactive correlation network graph using networkx + Plotly.
    Nodes = stocks, sized by degree centrality.
    Edges = correlations above threshold, width proportional to |correlation|.
    """
    if not HAS_PLOTLY or not HAS_NETWORKX:
        print("[VIZ] plotly/networkx not installed, skipping network graph")
        return None

    G = nx.Graph()
    N = len(tickers)

    for i in range(N):
        G.add_node(tickers[i])

    for i in range(N):
        for j in range(i + 1, N):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                G.add_edge(tickers[i], tickers[j], weight=abs(corr_val))

    if len(G.edges()) == 0:
        print(f"[VIZ] No edges above threshold {threshold}, lowering to 0.3")
        threshold = 0.3
        for i in range(N):
            for j in range(i + 1, N):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    G.add_edge(tickers[i], tickers[j], weight=abs(corr_val))

    # Spring layout
    pos = nx.spring_layout(G, k=2.5 / np.sqrt(N), iterations=80, seed=42)

    # Edge traces
    edge_x, edge_y, edge_widths = [], [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_widths.append(G.edges[edge]["weight"])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="rgba(255,255,255,0.15)"),
        hoverinfo="none",
    )

    # Node traces
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_degrees = [G.degree(n) for n in G.nodes()]

    # Color by sector if available
    if sector_map:
        from config import SECTOR_COLORS
        node_colors = [SECTOR_COLORS.get(sector_map.get(n, ""), COLORS["text_muted"])
                       for n in G.nodes()]
    else:
        node_colors = node_degrees

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=list(G.nodes()), textposition="top center",
        marker=dict(
            size=[d * 4 + 12 for d in node_degrees],
            color=node_colors,
            line=dict(width=2, color="rgba(255,255,255,0.6)"),
        ),
        hoverinfo="text",
        textfont=dict(size=8, color="white"),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=f"Correlation Network ({method.title()}, |corr| > {threshold})",
                       font=dict(color="white")),
            showlegend=False,
            template="plotly_dark",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor=COLORS["bg_dark"],
            plot_bgcolor=COLORS["surface"],
            width=800, height=700,
        ),
    )

    return fig


# =============================================================================
# Chart 9: Rolling Eigenvalue Animation (matplotlib)
# =============================================================================

def plot_rolling_eigenvalues(log_returns, window_months=6, step_months=1,
                             save_path=None):
    """
    Animated rolling eigenvalue spectrum showing how the distribution
    evolves over time. Returns the figure and animation object.
    """
    from src.rmt_analysis import build_correlation_matrix, eigenvalue_analysis_with_T
    from src.rmt_analysis import marchenko_pastur_pdf

    trading_days_per_month = 21
    window_days = window_months * trading_days_per_month
    step_days = step_months * trading_days_per_month

    rolling_data = []
    for start_idx in range(0, len(log_returns) - window_days, step_days):
        window = log_returns.iloc[start_idx:start_idx + window_days]
        if len(window) < 60:
            continue
        corr, N, T = build_correlation_matrix(window)
        eigenvalues, _, mp_bounds, _, _ = eigenvalue_analysis_with_T(corr, T)
        rolling_data.append((eigenvalues, mp_bounds, window.index[-1]))

    if not rolling_data:
        print("[VIZ] Not enough data for rolling eigenvalues")
        return None, None

    fig, ax = plt.subplots(figsize=(12, 6))

    def update(frame):
        ax.clear()
        eigenvalues, mp_bounds, date = rolling_data[frame]
        Q = mp_bounds["Q"]
        lam_plus = mp_bounds["lambda_plus"]

        ax.hist(eigenvalues, bins=min(30, len(eigenvalues)), density=True,
                alpha=0.5, color=COLORS["method_b"], edgecolor="none")
        x_mp = np.linspace(0, max(lam_plus + 1, eigenvalues.max()), 300)
        y_mp = marchenko_pastur_pdf(x_mp, Q)
        ax.plot(x_mp, y_mp, color=COLORS["gold"], linewidth=2.5)
        ax.axvline(lam_plus, color=COLORS["method_a"], linestyle="--", linewidth=1.5)
        ax.set_title(f"Eigenvalue Spectrum — {date.strftime('%Y-%m')}",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))
        ax.grid(True, alpha=0.2)

    try:
        import matplotlib.animation as animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(rolling_data),
            interval=600, repeat=True,
        )
        if save_path:
            ani.save(save_path, writer="pillow", fps=2)
        return fig, ani
    except Exception as e:
        print(f"[VIZ] Animation failed: {e}")
        update(0)
        return fig, None


# =============================================================================
# Chart 10: 3D Efficient Frontier Surface (Plotly)
# =============================================================================

def plot_3d_efficient_frontier(mu, cov, rf=RISK_FREE_RATE, n_points=50,
                               ef_returns=None, ef_vols=None, ef_sharpes=None,
                               save_path=None):
    """3D surface: X=volatility, Y=return, Z=Sharpe ratio with frontier line."""
    if not HAS_PLOTLY:
        print("[VIZ] plotly not installed, skipping 3D frontier")
        return None

    vols = np.linspace(0.05, 0.40, n_points)
    rets = np.linspace(0.0, 0.40, n_points)
    VOL, RET = np.meshgrid(vols, rets)
    SHARPE = np.where(VOL > 1e-10, (RET - rf) / VOL, 0)

    fig = go.Figure()

    # Sharpe surface
    fig.add_trace(go.Surface(
        x=VOL * 100, y=RET * 100, z=SHARPE,
        colorscale=[
            [0, COLORS["hm_neg"]],
            [0.3, COLORS["surface"]],
            [0.6, COLORS["surface_light"]],
            [1, COLORS["hm_pos"]],
        ],
        opacity=0.5,
        name="Sharpe Ratio Surface",
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white")),
    ))

    # Efficient frontier line
    if ef_returns is not None and ef_vols is not None and ef_sharpes is not None:
        fig.add_trace(go.Scatter3d(
            x=ef_vols * 100, y=ef_returns * 100, z=ef_sharpes,
            mode="lines+markers",
            name="Efficient Frontier",
            line=dict(color=COLORS["gold"], width=6),
            marker=dict(size=3, color=COLORS["gold"]),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg_dark"],
        scene=dict(
            xaxis_title="Volatility (%)",
            yaxis_title="Return (%)",
            zaxis_title="Sharpe Ratio",
            bgcolor=COLORS["surface"],
            xaxis=dict(gridcolor=COLORS["grid"]),
            yaxis=dict(gridcolor=COLORS["grid"]),
            zaxis=dict(gridcolor=COLORS["grid"]),
        ),
        title=dict(text="3D Sharpe Ratio Surface", font=dict(color="white")),
        width=850, height=700,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


# =============================================================================
# Chart 11: Weight Sankey Diagram (Plotly)
# =============================================================================

def plot_weight_sankey(weights_history, tickers, top_n=10, save_path=None):
    """
    Sankey diagram showing how portfolio weights flow across rebalancing periods.
    Tracks top_n most significant stocks.
    """
    if not HAS_PLOTLY:
        print("[VIZ] plotly not installed, skipping Sankey")
        return None

    n_periods = len(weights_history)
    if n_periods < 2:
        print("[VIZ] Need at least 2 rebalancing periods for Sankey")
        return None

    # Select top_n stocks by total weight across all periods
    total_weights = np.zeros(len(tickers))
    for w in weights_history:
        total_weights += w
    top_indices = np.argsort(total_weights)[::-1][:top_n]
    top_tickers = [tickers[i] for i in top_indices]

    # Build Sankey data
    labels = []
    source, target, value, link_color = [], [], [], []

    colors_cycle = [
        COLORS["method_b"], COLORS["method_a"], COLORS["method_c"],
        COLORS["gold"], COLORS["purple"], COLORS["orange"],
    ]

    for period in range(n_periods - 1):
        w_from = weights_history[period][top_indices]
        w_to = weights_history[period + 1][top_indices]

        for i, ticker in enumerate(top_tickers):
            label_from = f"{ticker} (P{period})"
            label_to = f"{ticker} (P{period + 1})"

            if label_from not in labels:
                labels.append(label_from)
            if label_to not in labels:
                labels.append(label_to)

            src_idx = labels.index(label_from)
            tgt_idx = labels.index(label_to)

            # Flow = min of weights (retained allocation)
            flow_val = min(w_from[i], w_to[i]) * 100
            if flow_val > 0.05:
                source.append(src_idx)
                target.append(tgt_idx)
                value.append(flow_val)
                link_color.append(colors_cycle[i % len(colors_cycle)] + "60")

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=12, thickness=18,
            line=dict(color="rgba(255,255,255,0.3)", width=0.5),
            label=labels,
            color=COLORS["surface_light"],
        ),
        link=dict(source=source, target=target, value=value, color=link_color),
    ))

    fig.update_layout(
        title=dict(text="Portfolio Weight Flow Across Rebalancing Periods",
                   font=dict(color="white")),
        template="plotly_dark",
        paper_bgcolor=COLORS["bg_dark"],
        font=dict(size=9, color="white"),
        width=900, height=600,
    )

    return fig


# =============================================================================
# Chart 12: Hierarchically Clustered Heatmap (seaborn)
# =============================================================================

def plot_clustered_heatmap(corr_matrix, save_path=None):
    """Correlation heatmap with hierarchical clustering dendrograms."""
    cmap = LinearSegmentedColormap.from_list(
        "rmt_cmap", [COLORS["hm_neg"], COLORS["hm_zero"], COLORS["hm_pos"]]
    )

    g = sns.clustermap(
        corr_matrix, method="ward", cmap=cmap, vmin=-1, vmax=1,
        figsize=(13, 13), dendrogram_ratio=0.12,
        cbar_pos=(0.02, 0.86, 0.03, 0.10),
        linewidths=0.3, linecolor=COLORS["bg_dark"],
        xticklabels=True, yticklabels=True,
        annot=False,
    )

    g.fig.patch.set_facecolor(COLORS["bg_dark"])
    g.ax_heatmap.set_facecolor(COLORS["surface"])
    g.ax_heatmap.tick_params(labelsize=7)
    g.fig.suptitle("Hierarchically Clustered Correlation Matrix", y=1.02,
                   fontsize=16, fontweight="bold")

    # Color tick labels by cluster
    if HAS_SCIPY_CLUSTER:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        dist = 1 - corr_matrix.abs()
        np.fill_diagonal(dist.values, 0)
        condensed = squareform(dist.values, checks=False)
        Z = linkage(condensed, method="ward")
        clusters = fcluster(Z, t=3, criterion="maxclust")
        cluster_colors = [COLORS["method_b"], COLORS["method_a"], COLORS["method_c"]]
        for i, label in enumerate(g.ax_heatmap.get_yticklabels()):
            if i < len(clusters):
                label.set_color(cluster_colors[clusters[i] - 1])

    if save_path:
        g.fig.savefig(save_path)
    return g.fig


# =============================================================================
# Chart 13: Animated Cumulative Return Race (Plotly)
# =============================================================================

def plot_cumulative_return_race(results, save_path=None):
    """Animated frame-by-frame cumulative return chart with play/pause controls."""
    if not HAS_PLOTLY:
        print("[VIZ] plotly not installed, skipping animated race")
        return None

    methods = [
        ("method_a", COLORS["method_a"], "Raw Markowitz"),
        ("method_b", COLORS["method_b"], "RMT-Denoised"),
        ("method_c", COLORS["method_c"], "Equal Weight"),
    ]

    # Normalize all to start at 1.0
    cum_data = {}
    max_len = 0
    for key, color, label in methods:
        cum = results[key]["cumulative"]
        if len(cum) > 0:
            cum_norm = cum / cum.iloc[0]
            cum_data[key] = (cum_norm.index, cum_norm.values, color, label)
            max_len = max(max_len, len(cum_norm))

    if max_len == 0:
        return None

    # Sample frames for performance (every N days)
    frame_step = max(1, max_len // 80)

    # Build frames
    frames = []
    for fi in range(frame_step, max_len, frame_step):
        frame_data = []
        for key, _, _ in methods:
            if key in cum_data:
                dates, values, c, l = cum_data[key]
                frame_data.append(go.Scatter(
                    x=dates[:fi], y=values[:fi],
                    mode="lines", name=l,
                    line=dict(color=c, width=3),
                ))
        frames.append(go.Frame(data=frame_data, name=str(fi)))

    # Initial empty traces
    fig = go.Figure()
    for key, color, label in methods:
        if key in cum_data:
            fig.add_trace(go.Scatter(
                x=[], y=[], mode="lines", name=label,
                line=dict(color=color, width=3),
            ))

    fig.frames = frames

    # Compute y-axis range
    y_max = max(max(d[1]) for d in cum_data.values()) * 1.1
    y_min = min(min(d[1]) for d in cum_data.values()) * 0.9

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Portfolio Value",
        yaxis=dict(range=[y_min, y_max]),
        template="plotly_dark",
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["surface"],
        title=dict(text="Cumulative Return Race", font=dict(color="white")),
        updatemenus=[dict(
            type="buttons",
            x=0.05, y=1.15,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=40, redraw=True),
                                      fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
            font=dict(color="white"),
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["baseline"],
        )],
        sliders=[dict(
            steps=[dict(
                args=[[str(i)], dict(frame=dict(duration=30, redraw=True),
                                     mode="immediate")],
                method="animate", label=str(i),
            ) for i in range(0, max_len, frame_step * 4)],
            currentvalue=dict(prefix="Day: ", font=dict(color="white")),
            len=0.9, x=0.05, y=0,
            bgcolor=COLORS["surface"],
            activebgcolor=COLORS["method_b"],
            bordercolor=COLORS["baseline"],
            font=dict(color="white"),
        )],
        width=900, height=550,
        legend=dict(font=dict(color="white")),
    )

    return fig


# =============================================================================
# Chart 14: Risk Contribution Bubble Chart (Plotly)
# =============================================================================

def plot_risk_contribution(weights, cov, tickers, method_name="Portfolio",
                           save_path=None):
    """
    Bubble chart: X=weight, Y=marginal risk contribution,
    bubble size=total risk contribution.
    """
    if not HAS_PLOTLY:
        print("[VIZ] plotly not installed, skipping risk contribution")
        return None

    port_vol = np.sqrt(weights @ cov @ weights)
    marginal_rc = (cov @ weights) / port_vol
    total_rc = weights * marginal_rc
    pct_rc = total_rc / port_vol * 100

    # Filter to significant positions
    mask = weights > 0.005
    sig_idx = np.where(mask)[0]
    sig_tickers = [tickers[i] for i in sig_idx]
    sig_weights = weights[mask] * 100
    sig_mrc = marginal_rc[mask]
    sig_total = pct_rc[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sig_weights, y=sig_mrc,
        mode="markers+text",
        text=sig_tickers, textposition="top center",
        marker=dict(
            size=np.abs(sig_total) * 15 + 8,
            color=sig_total,
            colorscale=[
                [0, COLORS["method_a"]],
                [0.5, COLORS["text_muted"]],
                [1, COLORS["method_b"]],
            ],
            line=dict(width=1, color="rgba(255,255,255,0.6)"),
            opacity=0.85,
            showscale=True,
            colorbar=dict(title="% Risk", font=dict(color="white")),
        ),
        textfont=dict(size=9, color="white"),
    ))

    # Equal risk contribution reference
    n_sig = len(sig_weights)
    if n_sig > 0:
        eq_rc = 100 / n_sig
        fig.add_hline(y=eq_rc, line_dash="dash", line_color=COLORS["text_muted"],
                      annotation_text=f"Equal Risk: {eq_rc:.1f}%",
                      annotation_font_color=COLORS["text_secondary"])

    fig.update_layout(
        title=dict(text=f"Risk Contribution Analysis — {method_name}",
                   font=dict(color="white")),
        xaxis_title="Portfolio Weight (%)",
        yaxis_title="Marginal Risk Contribution",
        template="plotly_dark",
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["surface"],
        width=800, height=550,
    )

    return fig


# =============================================================================
# Chart 15: Eigenvector Component Heatmap
# =============================================================================

def plot_eigenvector_heatmap(eigenvectors, eigenvalues, tickers, signal_mask,
                             sector_map=None, save_path=None):
    """
    Heatmap of top K eigenvector components per stock.
    Rows = stocks (grouped by sector), Columns = eigenvector components.
    """
    k_show = min(8, signal_mask.sum())
    if k_show == 0:
        print("[VIZ] No signal eigenvalues to plot")
        return None

    top_evecs = eigenvectors[:, :k_show]
    col_labels = [f"$\\lambda_{i+1}$={eigenvalues[i]:.2f}" for i in range(k_show)]

    df = pd.DataFrame(top_evecs, index=tickers, columns=col_labels)

    if sector_map:
        df["sector"] = df.index.map(sector_map)
        df = df.sort_values("sector")

    cmap = LinearSegmentedColormap.from_list(
        "evec", [COLORS["hm_neg"], COLORS["surface"], COLORS["hm_pos"]]
    )

    fig, ax = plt.subplots(figsize=(10, max(12, len(tickers) * 0.38)))
    data_to_plot = df.drop(columns=["sector"]) if "sector" in df.columns else df

    sns.heatmap(data_to_plot, cmap=cmap, center=0, ax=ax,
                linewidths=0.3, linecolor=COLORS["bg_dark"],
                cbar_kws={"shrink": 0.5, "label": "Component"})

    # Color tick labels by sector
    if sector_map and "sector" in df.columns:
        from config import SECTOR_COLORS
        for i, label in enumerate(ax.get_yticklabels()):
            sector = df["sector"].iloc[i] if i < len(df) else ""
            label.set_color(SECTOR_COLORS.get(sector, COLORS["text_primary"]))

    ax.set_title("Signal Eigenvector Components per Stock", fontsize=14, pad=15)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# Chart 16: Sector Correlation Matrix (Plotly)
# =============================================================================

def plot_sector_correlation(corr_matrix, tickers, sector_map, save_path=None):
    """Aggregated sector-level correlation matrix."""
    if not HAS_PLOTLY:
        print("[VIZ] plotly not installed, skipping sector correlation")
        return None

    # Get unique sectors in order
    sector_tickers = {}
    for t in tickers:
        s = sector_map.get(t, "Unknown")
        sector_tickers.setdefault(s, []).append(t)

    sectors = sorted(sector_tickers.keys())
    n_sectors = len(sectors)
    sector_corr = np.zeros((n_sectors, n_sectors))

    for i, s1 in enumerate(sectors):
        for j, s2 in enumerate(sectors):
            st1 = sector_tickers[s1]
            st2 = sector_tickers[s2]
            # Only use tickers that exist in the correlation matrix
            st1 = [t for t in st1 if t in corr_matrix.columns]
            st2 = [t for t in st2 if t in corr_matrix.columns]
            if st1 and st2:
                sector_corr[i, j] = corr_matrix.loc[st1, st2].values.mean()

    fig = go.Figure(go.Heatmap(
        z=sector_corr, x=sectors, y=sectors,
        colorscale=[
            [0, COLORS["hm_neg"]],
            [0.5, COLORS["surface"]],
            [1, COLORS["hm_pos"]],
        ],
        text=np.round(sector_corr, 2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=11, color="white"),
    ))

    fig.update_layout(
        title=dict(text="Sector-Level Correlation Matrix", font=dict(color="white")),
        template="plotly_dark",
        paper_bgcolor=COLORS["bg_dark"],
        width=750, height=700,
    )

    return fig


# =============================================================================
# Chart 17: Composite Dashboard (matplotlib)
# =============================================================================

def plot_dashboard(results, metrics_df, eigenvalues, mp_bounds,
                   corr_raw, corr_denoised, save_path=None):
    """
    Single composite dashboard figure with 6 panels:
    - Top-left:     Key metrics tiles
    - Top-center:   Mini eigenvalue spectrum
    - Top-right:    Mini correlation heatmap
    - Bottom-left:  Cumulative returns
    - Bottom-center: Rolling Sharpe
    - Bottom-right: Drawdown
    """
    fig = plt.figure(figsize=(24, 14), facecolor=COLORS["bg_dark"])
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.28,
                           left=0.04, right=0.96, top=0.91, bottom=0.05)

    fig.suptitle("RMT Portfolio Optimization Dashboard", fontsize=22,
                 fontweight="bold", color=COLORS["text_primary"], y=0.97)

    # --- Panel 1: Metrics tiles ---
    ax_tiles = fig.add_subplot(gs[0, 0])
    ax_tiles.axis("off")
    ax_tiles.set_facecolor(COLORS["surface"])

    tile_data = []
    for method in metrics_df.index:
        row = metrics_df.loc[method]
        tile_data.append({
            "name": method,
            "sharpe": row.get("Sharpe Ratio", "N/A"),
            "return": row.get("Annual Return", "N/A"),
            "vol": row.get("Annual Volatility", "N/A"),
            "dd": row.get("Max Drawdown", "N/A"),
        })

    tile_colors = [COLORS["method_a"], COLORS["method_b"], COLORS["method_c"]]
    for i, (td, tc) in enumerate(zip(tile_data, tile_colors)):
        y_start = 0.95 - i * 0.33
        _frosted_box(ax_tiles, f"● {td['name']}", 0.05, y_start,
                     fontsize=11, color=tc)
        _frosted_box(ax_tiles,
                     f"  Sharpe: {td['sharpe']}  |  Ret: {td['return']}\n"
                     f"  Vol: {td['vol']}  |  MaxDD: {td['dd']}",
                     0.05, y_start - 0.10, fontsize=9, color=COLORS["text_secondary"])

    # --- Panel 2: Mini eigenvalue spectrum ---
    ax_eig = fig.add_subplot(gs[0, 1])
    ax_eig.set_facecolor(COLORS["surface"])
    n_bins = min(30, len(eigenvalues))
    ax_eig.hist(eigenvalues, bins=n_bins, density=True, alpha=0.5,
                color=COLORS["method_b"], edgecolor="none")
    x_mp = np.linspace(0, mp_bounds["lambda_plus"] + 0.5, 200)
    from src.rmt_analysis import marchenko_pastur_pdf
    y_mp = marchenko_pastur_pdf(x_mp, mp_bounds["Q"])
    ax_eig.plot(x_mp, y_mp, color=COLORS["gold"], linewidth=2)
    ax_eig.axvline(mp_bounds["lambda_plus"], color=COLORS["method_a"],
                   linestyle="--", linewidth=1)
    ax_eig.set_title("Eigenvalue Spectrum", fontsize=12, pad=8)
    ax_eig.grid(True, alpha=0.2)

    # --- Panel 3: Mini correlation heatmap ---
    ax_hm = fig.add_subplot(gs[0, 2])
    ax_hm.set_facecolor(COLORS["surface"])
    n_show = min(15, corr_raw.shape[0])
    cmap = LinearSegmentedColormap.from_list(
        "rmt", [COLORS["hm_neg"], COLORS["hm_zero"], COLORS["hm_pos"]]
    )
    im = ax_hm.imshow(corr_raw.iloc[:n_show, :n_show].values, cmap=cmap,
                      vmin=-1, vmax=1, aspect="auto")
    ax_hm.set_title("Raw Correlation", fontsize=12, pad=8)
    ax_hm.set_xticks([])
    ax_hm.set_yticks([])

    # --- Panel 4: Cumulative returns ---
    ax_cum = fig.add_subplot(gs[1, 0])
    ax_cum.set_facecolor(COLORS["surface"])
    methods = [
        ("method_a", COLORS["method_a"], "Raw Markowitz"),
        ("method_b", COLORS["method_b"], "RMT-Denoised"),
        ("method_c", COLORS["method_c"], "Equal Weight"),
    ]
    for key, color, label in methods:
        cum = results[key]["cumulative"]
        if len(cum) > 0:
            cum_norm = cum / cum.iloc[0]
            ax_cum.plot(cum_norm.index, cum_norm.values, color=color,
                        linewidth=1.8, label=label, alpha=0.85)
    ax_cum.axhline(1.0, color=COLORS["text_muted"], linestyle=":", linewidth=0.8, alpha=0.5)
    ax_cum.set_title("Cumulative Returns", fontsize=12, pad=8)
    ax_cum.legend(fontsize=8, framealpha=0.3, edgecolor=COLORS["baseline"])
    ax_cum.grid(True, alpha=0.2)

    # --- Panel 5: Rolling Sharpe ---
    ax_sharpe = fig.add_subplot(gs[1, 1])
    ax_sharpe.set_facecolor(COLORS["surface"])
    for key, color, label in methods:
        daily = results[key]["daily"]
        if len(daily) > 63:
            rolling_ret = daily.rolling(63).mean() * 252
            rolling_vol = daily.rolling(63).std() * np.sqrt(252)
            rs = (rolling_ret - RISK_FREE_RATE) / rolling_vol
            rs_clean = rs.dropna()
            if len(rs_clean) > 0:
                ax_sharpe.plot(rs_clean.index, rs_clean.values, color=color,
                               linewidth=1.2, label=label, alpha=0.8)
    ax_sharpe.axhline(0, color=COLORS["text_muted"], linestyle="--", linewidth=0.8, alpha=0.5)
    ax_sharpe.set_title("Rolling Sharpe (63d)", fontsize=12, pad=8)
    ax_sharpe.grid(True, alpha=0.2)

    # --- Panel 6: Drawdown ---
    ax_dd = fig.add_subplot(gs[1, 2])
    ax_dd.set_facecolor(COLORS["surface"])
    for key, color, label in methods:
        dd = results[key]["drawdown"]
        if len(dd) > 0:
            ax_dd.fill_between(dd.index, 0, dd.values * 100, alpha=0.12, color=color)
            ax_dd.plot(dd.index, dd.values * 100, color=color, linewidth=1.2,
                       label=label, alpha=0.85)
    ax_dd.set_title("Drawdown (%)", fontsize=12, pad=8)
    ax_dd.legend(fontsize=8, framealpha=0.3, edgecolor=COLORS["baseline"])
    ax_dd.grid(True, alpha=0.2)

    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    setup_dark_style()
    print("Viz suite loaded. Functions available:")
    for name in dir():
        if name.startswith("plot_"):
            print(f"  - {name}")
