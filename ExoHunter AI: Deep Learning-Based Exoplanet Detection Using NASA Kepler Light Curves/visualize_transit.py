"""
Light Curve Classifier — Transit Visualisation
===============================================
Plots a single normalised light curve and highlights
the deepest flux dip as the candidate transit point.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Transit visualisation
# ---------------------------------------------------------------------------

def plot_transit(curve, index=0, save_path=None):
    """
    Plot a single light curve and mark the deepest dip.

    Parameters
    ----------
    curve     : array-like   1-D flux time series (already normalised)
    index     : int          Sample index — used only for the plot title
    save_path : str | None   If given, save figure to this path
    """

    curve    = np.array(curve).ravel()          # ensure 1-D
    dip_idx  = int(np.argmin(curve))            # index of minimum flux
    dip_val  = curve[dip_idx]

    # Transit depth = drop from median to minimum  (more robust than max→min)
    median_flux   = np.median(curve)
    transit_depth = median_flux - dip_val

    fig, ax = plt.subplots(figsize=(12, 5))

    # ── Light curve ───────────────────────────────────────────────────────
    ax.plot(
        curve,
        color="steelblue",
        linewidth=0.9,
        alpha=0.85,
        label="Normalised Flux",
    )

    # ── Median baseline ───────────────────────────────────────────────────
    ax.axhline(
        median_flux,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label=f"Median flux  ({median_flux:.4f})",
    )

    # ── Transit dip marker ────────────────────────────────────────────────
    ax.scatter(
        dip_idx,
        dip_val,
        s=120,
        color="red",
        zorder=5,
        label=f"Deepest dip  (t={dip_idx},  flux={dip_val:.4f})",
    )

    # Vertical guide line to dip
    ax.axvline(
        dip_idx,
        color="red",
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
    )

    # ── Transit depth annotation ──────────────────────────────────────────
    ax.annotate(
        f"Transit depth\n{transit_depth:.4f}",
        xy=(dip_idx, dip_val),
        xytext=(dip_idx + len(curve) * 0.04, dip_val + transit_depth * 0.4),
        fontsize=8,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
    )

    ax.set_title(
        f"Potential Exoplanet Transit — Sample {index}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Normalised Flux")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {save_path}")

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Multi-curve overview  (optional: show first N samples in a grid)
# ---------------------------------------------------------------------------

def plot_transit_grid(X, n=6, cols=3, save_path=None):
    """
    Plot the first `n` light curves in a grid for a quick dataset overview.

    Parameters
    ----------
    X         : array-like   Shape (N, seq_len) or (N, seq_len, 1)
    n         : int          Number of samples to show
    cols      : int          Columns in the grid
    save_path : str | None   Optional save path
    """

    X   = np.array(X)
    if X.ndim == 3:
        X = X[:, :, 0]                          # drop feature dim if present

    n    = min(n, len(X))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    axes = np.array(axes).ravel()

    for i in range(n):
        curve   = X[i]
        dip_idx = int(np.argmin(curve))

        axes[i].plot(curve, color="steelblue", linewidth=0.8, alpha=0.85)
        axes[i].scatter(dip_idx, curve[dip_idx], s=60, color="red", zorder=5)
        axes[i].axhline(np.median(curve), color="gray", linestyle="--",
                        linewidth=0.7, alpha=0.5)
        axes[i].set_title(f"Sample {i}", fontsize=9)
        axes[i].set_xlabel("Time Step", fontsize=7)
        axes[i].set_ylabel("Flux", fontsize=7)
        axes[i].tick_params(labelsize=7)
        axes[i].grid(linestyle="--", alpha=0.25)

    # Hide any unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Light Curve Overview — First Samples", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grid plot saved → {save_path}")

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Single transit plot
    plot_transit(
        curve=X[0],
        index=0,
        save_path="transit_sample_0.png",
    )

    # Grid overview of first 6 samples
    plot_transit_grid(
        X=X,
        n=6,
        cols=3,
        save_path="transit_grid.png",
    )
